import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import albumentations as A


def euclidean_distance(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


class DataLoader:
    def __init__(self, images_dir, augment=False):
        self.images_dir = images_dir
        self.augment = augment
        if self.augment:
            self.augmenter = A.Compose([
                A.Rotate(limit=20, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5)
            ], keypoint_params=A.KeypointParams(format='xy'))

    def augment_data(self, img, coords):
        augmented = self.augmenter(image=img, keypoints=coords)
        img_aug = augmented['image']
        coords_aug = augmented['keypoints']

        coords_aug = [(min(max(x, 0), img_aug.shape[1] - 1), min(max(y, 0), img_aug.shape[0] - 1)) for x, y in
                      coords_aug]

        return img_aug, coords_aug

    def load_data(self):
        images = []
        coords = []
        files = os.listdir(self.images_dir)
        print(f"Fichiers trouvés : {files}")

        for filename in files:
            if filename.endswith('.png'):
                img_name = filename
                txt_name = filename.replace('.png', '.txt')
                img_path = os.path.join(self.images_dir, img_name)
                txt_path = os.path.join(self.images_dir, txt_name)

                if not os.path.exists(img_path) or not os.path.exists(txt_path):
                    continue

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Lire l'image en noir et blanc
                img_resized = cv2.resize(img, (1024, 1024))

                with open(txt_path, 'r') as file:
                    coord_list = file.read().strip().split('),(')
                    coords_for_image = []
                    for coord in coord_list:
                        coord = coord.replace('(', '').replace(')', '')
                        try:
                            x, y = map(float, coord.split(','))
                            x = min(max(x, 0), img_resized.shape[1] - 1)
                            y = min(max(y, 0), img_resized.shape[0] - 1)
                            coords_for_image.append((x, y))
                        except ValueError:
                            continue

                if self.augment:
                    img_resized, coords_for_image = self.augment_data(img_resized, coords_for_image)

                images.append(img_resized)
                coords.extend(coords_for_image)

        images = np.expand_dims(np.array(images, dtype="float32") / 255.0, axis=-1)  # Ajouter une dimension pour le canal
        coords = np.array(coords, dtype="float32").reshape(-1, 2)

        return images, coords


class ObjectDetectionModel:
    def __init__(self, input_shape=(1024, 1024, 1)):  # Modifier le shape pour une seule chaîne de couleur
        self.input_shape = input_shape
        self.model = None

    def build_model(self):
        base_model = tf.keras.applications.EfficientNetB5(
            input_shape=self.input_shape,
            include_top=False,
            weights=None  # Pas de pré-entraînement avec ImageNet car ce sont des images en noir et blanc
        )

        base_model.trainable = False

        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(128, activation='relu')(x)  # Utilisation d'une taille fixe de couche dense
        x = layers.Dropout(0.3)(x)  # Utilisation d'un taux de dropout fixe
        output = layers.Dense(2)(x)

        model = models.Model(inputs=base_model.input, outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Taux d'apprentissage fixe
            loss='mse',
            metrics=[euclidean_distance]
        )

        return model

    def compile_model(self):
        if self.model is None:
            self.model = self.build_model()

    def train(self, images, coords, epochs=200, batch_size=4, validation_split=0.2):
        if len(images) == 0 or len(coords) == 0:
            raise ValueError("Aucune donnée n'a été chargée pour l'entraînement.")

        checkpoint_callback = ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            save_weights_only=False,
            verbose=1
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(
            images, coords,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[checkpoint_callback, early_stopping]
        )

        self._plot_training(history)

    def _plot_training(self, history):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Perte d\'entraînement')
        plt.plot(history.history['val_loss'], label='Perte de validation')
        plt.title('Perte au cours des époques')
        plt.xlabel('Époque')
        plt.ylabel('Perte')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['euclidean_distance'], label='Distance euclidienne entraînement')
        plt.plot(history.history['val_euclidean_distance'], label='Distance euclidienne validation')
        plt.title('Distance euclidienne au cours des époques')
        plt.xlabel('Époque')
        plt.ylabel('Distance euclidienne')
        plt.legend()

        plt.show()

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects={'euclidean_distance': euclidean_distance})


# Exemple d'utilisation
if __name__ == "__main__":
    images_dir = '30SampleXY/'
    print(f"Répertoire utilisé : {images_dir}")
    data_loader = DataLoader(images_dir, augment=True)
    images, coords = data_loader.load_data()

    if len(images) == 0 or len(coords) == 0:
        print("Erreur : Aucune donnée d'entraînement trouvée.")
        exit(1)

    model = ObjectDetectionModel()
    model.compile_model()
    model.train(images, coords, epochs=10000, batch_size=4)

    model_save_path = 'saved_model.keras'
    model.save_model(model_save_path)
