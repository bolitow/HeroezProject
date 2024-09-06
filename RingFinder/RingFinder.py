import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import requests
import csv
import os


# Fonction pour récupérer les données depuis l'API
def fetch_data_from_api(url, period_key="s22_s1"):
    response = requests.get(url, verify=False)

    if response.status_code == 200:
        data = response.json()
        filtered_data = [item for item in data if item.get('periodKey') == period_key]

        if filtered_data:
            return filtered_data
        else:
            print(f"Aucune donnée ne correspond à '{period_key}'.")
            return None
    else:
        print(f"Erreur lors de la requête: {response.status_code}")
        return None


# Fonction pour sauvegarder les données dans un fichier CSV avec un nom dynamique
def save_data_to_csv(data, map_name):
    # Déterminer le nom du fichier en fonction du nom de la carte
    if map_name == "mp_rr_tropic_island_mu2":
        filename = 'StormPoint.csv'
    else:
        filename = 'WorldEdge.csv'

    # Sauvegarder les données dans le fichier CSV
    keys = data[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)
    print(f"Les données filtrées ont été enregistrées dans '{filename}'.")


# Fonction pour extraire les coordonnées x et y des points
def extract_points(data):
    return [(item['x'], item['y']) for item in data]


# Fonction pour créer une heatmap à partir des points
def create_heatmap(points, img_path, output_dir, map_name, img_size=(4096, 4096), sigma=30, threshold=0.0001):
    # Charger l'image de la carte
    carte = Image.open(img_path)

    largeur, hauteur = img_size
    heatmap_data = np.zeros((hauteur, largeur))

    # Si 'points' est une liste de tuples, convertissons-les en tableau NumPy pour l'optimisation
    points = np.array(points)

    # Réduire les coordonnées des points par un facteur de 4
    X = points[:, 0] // 4
    Y = points[:, 1] // 4

    # Filtrer les points valides qui sont dans la plage de la carte
    valid_indices = (X >= 0) & (X < largeur) & (Y >= 0) & (Y < hauteur)
    X_valid = X[valid_indices]
    Y_valid = Y[valid_indices]

    # Incrémenter les cellules correspondantes dans la heatmap en une seule opération
    np.add.at(heatmap_data, (Y_valid, X_valid), 1)

    # Appliquer un flou gaussien pour lisser la heatmap
    heatmap_data = gaussian_filter(heatmap_data, sigma=sigma)

    # Vérifier si la heatmap est non vide avant d'ajuster les valeurs
    if np.any(heatmap_data):
        heatmap_data[heatmap_data < threshold] = np.nan  # Rendre les valeurs sous le seuil transparentes
    else:
        print("La heatmap est vide, vérifiez les points d'entrée.")
        return

    # Nommer le fichier de sortie en fonction de la carte
    output_filename = f"heatmap_{map_name}.png"
    output_path = os.path.join(output_dir, output_filename)

    # Afficher et enregistrer la heatmap
    display_heatmap(heatmap_data, carte, threshold, output_path)



# Fonction pour afficher et enregistrer la heatmap
def display_heatmap(heatmap_data, carte, threshold, output_path):
    # Créer une colormap personnalisée
    colors = ['green', 'red']
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    # Définir les limites pour l'affichage de la heatmap
    vmin = threshold
    vmax = np.nanpercentile(heatmap_data[~np.isnan(heatmap_data)], 95)

    # Définir la taille de la figure
    dpi = 600
    figsize = (4096 / dpi, 4096 / dpi)

    plt.figure(figsize=figsize, dpi=dpi)

    # Afficher la heatmap superposée à l'image
    plt.imshow(carte, aspect='auto', alpha=1)  # Image de fond avec pleine opacité
    sns.heatmap(heatmap_data, cmap=custom_cmap, alpha=0.4, zorder=2, cbar=False, vmin=vmin, vmax=vmax,
                mask=np.isnan(heatmap_data))

    plt.axis('off')

    # Enregistrer l'image
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    print(f"Heatmap sauvegardée sous {output_path}")
    plt.show()


def select_map():
    name = input("Choississez le nom d'une map 1. Stormpoint 2.WorldEdge")
    if name == '1':
        map_name = 'mp_rr_tropic_island_mu2'
        return map_name
    if name == '2':
        map_name = "mp_rr_desertlands_hu"
        return map_name
    else:
        print('error')
        exit()


# Fonction principale pour orchestrer l'exécution
def main():
    map_name = select_map()
    # URL de l'API
    api_url = f"https://apexlegendsstatus.com/tournament/atls/getEndZones?map={map_name}"
    print(api_url)

    # Récupérer les données
    data = fetch_data_from_api(api_url)

    if data:
        # Sauvegarder les données dans le fichier CSV avec le nom approprié
        save_data_to_csv(data, map_name)

        # Extraire les points
        points = extract_points(data)

        # Chemin de l'image et dossier de sortie
        img_path = f"{map_name}.png"
        output_dir = "."

        # Créer et afficher la heatmap
        create_heatmap(points, img_path, output_dir, map_name)


# Appel de la fonction principale
if __name__ == "__main__":
    main()
