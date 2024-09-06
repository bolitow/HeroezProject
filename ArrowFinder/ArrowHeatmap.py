import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import os
import pandas as pd


# Fonction pour extraire les coordonnées x et y des clusters à partir du fichier CSV
def extract_points_from_csv(csv_filename):
    data = pd.read_csv(csv_filename)
    points = [(row['centre_x'], row['centre_y']) for _, row in data.iterrows()]
    return points


# Fonction pour créer une heatmap à partir des points et la superposer sur une image
def create_heatmap(points, img_path, output_dir, output_filename, img_size=(1920, 1080), sigma=30, threshold=0.002):
    # Charger l'image de la carte (fond)
    carte = Image.open(img_path)

    largeur, hauteur = img_size
    heatmap_data = np.zeros((hauteur, largeur))

    # Remplir la grille avec les points
    for x, y in points:
        if 0 <= x < largeur and 0 <= y < hauteur:
            heatmap_data[int(y), int(x)] += 1

    # Appliquer un flou gaussien pour lisser la heatmap
    heatmap_data = gaussian_filter(heatmap_data, sigma=sigma)

    # Vérifier si la heatmap est non vide avant d'ajuster les valeurs
    if np.any(heatmap_data):
        heatmap_data[heatmap_data < threshold] = np.nan  # Rendre les valeurs sous le seuil transparentes
    else:
        print("La heatmap est vide, vérifiez les points d'entrée.")
        return

    # Afficher et enregistrer la heatmap
    display_heatmap(heatmap_data, carte, threshold, os.path.join(output_dir, output_filename))


# Fonction pour afficher et enregistrer la heatmap
def display_heatmap(heatmap_data, carte, threshold, output_path):
    # Créer une colormap personnalisée
    colors = ['green', 'red']
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    # Définir les limites pour l'affichage de la heatmap
    vmin = threshold
    vmax = np.nanpercentile(heatmap_data[~np.isnan(heatmap_data)], 95)

    # Définir la taille de la figure
    dpi = 300
    figsize = (1920 / dpi, 1080 / dpi)

    plt.figure(figsize=figsize, dpi=dpi)

    # Afficher la heatmap superposée à l'image sans la barre de couleur
    plt.imshow(carte, aspect='auto', alpha=1)  # Image de fond avec pleine opacité
    sns.heatmap(heatmap_data, cmap=custom_cmap, square=False, alpha=0.3, zorder=2, #alpha=05 plus le chiffre haut moins c'est transparent
                cbar=False, vmin=vmin, vmax=vmax, mask=np.isnan(heatmap_data))  # cbar=False pour supprimer l'échelle

    plt.axis('off')

    # Enregistrer l'image
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    print(f"Heatmap sauvegardée sous {output_path}")
    plt.close()


# Fonction principale
def main():
    csv_filename = "clusters_data.csv"  # Fichier CSV contenant les clusters
    img_path = "map_heroez.png"  # Image de fond
    output_dir = "."  # Dossier de sortie
    output_filename = "arrow_heatmap.png"  # Nom de l'image de sortie

    # Extraire les points (centre_x, centre_y) à partir du CSV
    points = extract_points_from_csv(csv_filename)

    # Créer et afficher la heatmap superposée sur la carte
    create_heatmap(points, img_path, output_dir, output_filename, img_size=(1920, 1080), sigma=25)


if __name__ == "__main__":
    main()
