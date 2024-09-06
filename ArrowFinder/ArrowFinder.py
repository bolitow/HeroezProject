import os
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist
import csv
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor

couleur_cible = np.array([255, 255, 28])
tol = 60
distance_max = 4
limite_x = 1466
limite_x2 = 452
marge = 10
taille_min_carre = 20
taille_max_carre = 40
seuil_voisins = 10


def est_proche(couleur, cible, tol):
    return np.linalg.norm(couleur - cible, axis=-1) <= tol


def detecter_pixels_proches(image_array):
    mask = est_proche(image_array, couleur_cible, tol)
    coords = np.argwhere(mask)
    coords = [(x, y) for y, x in coords if limite_x2 <= x <= limite_x]
    return coords


def regrouper_en_clusters(coords):
    coords_array = np.array(coords)
    clustering = DBSCAN(eps=distance_max, min_samples=1).fit(coords_array)
    clusters = [coords_array[clustering.labels_ == i] for i in set(clustering.labels_)]
    return clusters


def exclure_clusters_proches(clusters, distance_seuil=10, taille_seuil=100):
    """Exclut les clusters s'ils sont à moins de distance_seuil pixels et qu'au moins un des clusters a plus de
    taille_seuil éléments."""
    clusters_a_exclure = set()

    # Parcourir chaque pair de clusters
    for i, cluster1 in enumerate(clusters):
        for j, cluster2 in enumerate(clusters):
            if i >= j:  # Éviter les comparaisons redondantes ou avec soi-même
                continue

            # Calculer la distance minimale entre deux clusters
            distances = cdist(cluster1, cluster2)
            min_distance = np.min(distances)

            # Vérifier si les clusters sont proches et si l'un d'eux dépasse la taille seuil
            if min_distance <= distance_seuil and (len(cluster1) > taille_seuil or len(cluster2) > taille_seuil):
                clusters_a_exclure.add(i)
                clusters_a_exclure.add(j)

    # Exclure les clusters identifiés
    clusters_restants = [cluster for i, cluster in enumerate(clusters) if i not in clusters_a_exclure]

    return clusters_restants


def tracer_carre_rouge_avec_filtre(clusters, image):
    """Trace un carré rouge autour des clusters qui respectent les critères de taille et retourne les clusters restants."""
    draw = ImageDraw.Draw(image)
    clusters_restants = []

    for cluster in clusters:
        # Calculer les limites du rectangle englobant pour chaque cluster
        x_min = min([x for x, y in cluster]) - marge
        y_min = min([y for x, y in cluster]) - marge
        x_max = max([x for x, y in cluster]) + marge
        y_max = max([y for x, y in cluster]) + marge

        # Calculer la largeur et la hauteur du carré
        largeur_carre = x_max - x_min
        hauteur_carre = y_max - y_min

        # Vérifier si le carré respecte les critères de taille
        if taille_min_carre <= largeur_carre <= taille_max_carre and taille_min_carre <= hauteur_carre <= taille_max_carre:
            # Dessiner un rectangle rouge autour de chaque cluster qui respecte les critères
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=4)
            clusters_restants.append(cluster)  # Ajouter les clusters valides aux clusters restants

    return clusters_restants


def enregistrer_clusters_csv(clusters, image_name, csv_filename):
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for cluster in clusters:
            centre_x, centre_y = np.mean(cluster, axis=0).astype(int)
            writer.writerow([image_name, centre_x, centre_y, len(cluster)])


def traiter_image(image_path, output_dir, csv_filename):
    """Charge une image, détecte les clusters, trace des carrés rouges et enregistre dans un CSV."""
    image = Image.open(image_path)
    pixels = np.array(image)

    # Détecter les pixels proches de la couleur cible
    coords = detecter_pixels_proches(pixels)

    # Regrouper les pixels en clusters
    clusters = regrouper_en_clusters(coords)

    # Exclure les clusters trop proches et ceux qui dépassent le seuil de taille
    clusters = exclure_clusters_proches(clusters, distance_seuil=10, taille_seuil=100)

    # Tracer les carrés rouges autour des clusters restants et ne conserver que ces clusters
    clusters_restants = tracer_carre_rouge_avec_filtre(clusters, image)

    # Sauvegarder l'image avec les rectangles dans le répertoire de sortie
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"output_{image_name}")
    image.save(output_path)

    # Enregistrer les informations des clusters restants dans le CSV
    enregistrer_clusters_csv(clusters_restants, image_name, csv_filename)


def traiter_repertoire(input_dir, output_dir, csv_filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["nom_sample", "centre_x", "centre_y", "nombre_elements"])
    image_files = [f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".jfif"))]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(traiter_image, os.path.join(input_dir, img), output_dir, csv_filename) for img in
                   image_files]
        for future in futures:
            future.result()


# Appeler la fonction pour traiter un répertoire et enregistrer dans un CSV
input_directory = "Test/"
output_directory = "Test_output/"
csv_filename = "clusters_data.csv"
traiter_repertoire(input_directory, output_directory, csv_filename)
