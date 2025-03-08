import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import os
from fpdf import FPDF
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.interpolate import griddata
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

# 1-Collecte, compréhension et audit de la qualité des données
# Source : https://www.kaggle.com/code/vikasukani/forest-fires-eda-solution-data-analysis/input
# Chargement des données dans un dataframe
df = pd.read_csv("forestfires.csv")

# Affichage des premières lignes
print(df.head())

print("##################################################################################")

# Affichage des informations sur le dataframe
print(df.info())

print("##################################################################################")

# Conversion en float de la colonne "area" et contrôle des premières lignes de la colonne convertie "area" du dataframe
df["area"] = df["area"].str.replace(",", ".").astype(float)
print(f"Premières valeurs de la colonne area converties en float: ")
print(df["area"].head())

# Conversion en category des colonnes "month" et "day"
df["month"] = df["month"].astype("category")
df["day"] = df["day"].astype("category")
print(df.info())

print("##################################################################################")

# Affichage du nombre de valeurs manquantes pour chaque colonne
missing_counts = np.sum(df.isnull(), axis=0)

for col, count in zip(df.columns, missing_counts):
    print(f"Colonne {col}: {count} valeur(s) manquante(s)")

print("##################################################################################")

# Recherche des valeurs distinctes et de leur nombre pour les colonnes "month" et "day"
columns_to_check = ["month", "day"]

for column in columns_to_check:
    unique_values = np.unique(df[column])
    unique_count = len(unique_values)
    print(f"Colonne {column}:")
    print(f"  Valeurs distinctes: {unique_values}")
    print(f"  Nombre de valeurs distinctes: {unique_count}\n")

print("##################################################################################")

# Recherche de la valeur minimale et maximale des indices du Fire Weather Index (FWI), du vent, des précipitations et de la surface brûlée
columns_to_check = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"]

result_minmax = df[columns_to_check].agg(["min", "max"])
print(result_minmax)

print("##################################################################################")

# Recherche des valeurs des indices du FWI au-dessus de la plage théorique de valeurs

# Plage théorique de valeurs pour chaque indice du FWI selon https://confluence.ecmwf.int/display/CEMS/User+Guide
theoretical_ranges = {
    "FFMC": (0, 101),  # FFMC ranges from 0 to 101, where higher values indicate drier and more easily ignitable fuels.
    "DMC": (0, 1000),  # DMC ranges from 0 to 1000, with higher values indicating drier conditions.
    "DC": (0, 1000),  # DC ranges from 0 to 1000, with higher values indicating drier conditions.
    "ISI": (0, 50)  # ISI ranges from 0 to 50, with higher values indicating a faster fire spread potential.
}

# Liste des colonnes à vérifier
columns_to_check = ["FFMC", "DMC", "DC", "ISI"]

# Vérification des valeurs au-dessus de la plage théorique pour les indices du FWI
for column in theoretical_ranges:
    min_val, max_val = theoretical_ranges[column]

    # Recherche des lignes dont les valeurs sont au-dessus de la plage théorique
    above_max = df[df[column] > max_val]
    if not above_max.empty:
        print(f"\nLignes où la valeur de {column} dépasse la plage théorique : {max_val}")
        print(f"Nombre de lignes : {above_max.shape[0]}")
        print("Indices des lignes dépassant la plage théorique :")
        print(above_max.index.tolist())
    else:
        print(f"\nAucune ligne pour {column} qui dépasse la plage théorique.")

print("##################################################################################")

# Recherche de lignes dont les valeurs ont un type non conforme à leur colonne

# Dictionnaire du type attendu pour chaque colonne
expected_types = {
    "FFMC": float,
    "DMC": float,
    "DC": float,
    "ISI": float,
    "temp": float,
    "RH": int,
    "wind": float,
    "rain": float,
    "area": float
}

# Initialisation d'un drapeau pour indiquer s'il y a des valeurs non conformes
has_non_conforming_rows = False

# Boucle pour vérifier les types dans chaque colonne
for column, expected_type in expected_types.items():
    # Filtrer les lignes où le type n'est pas conforme
    non_conforming_rows = df[~df[column].apply(lambda x: isinstance(x, expected_type))]

    if not non_conforming_rows.empty:
        has_non_conforming_rows = True
        print(f"Colonnes avec valeurs non conformes dans '{column}': ")
        print(non_conforming_rows, "\n")

# Si aucune ligne non conforme n'a été trouvée
if not has_non_conforming_rows:
    print("Aucune ligne trouvée de type non conforme avec la grandeur de la colonne. ")

print("##################################################################################")

# Recherche des doublons et affichage des paires de lignes en doublon
duplicates = df[df.duplicated(keep=False)]  # Conserver toutes les occurrences des doublons

if duplicates.empty:
    print("Aucun doublon trouvé. ")
else:
    # Récupérer les indices des doublons
    duplicate_indices = duplicates.index.tolist()

    to_drop = []

    # Comparer chaque doublon avec les autres doublons
    for i in range(len(duplicate_indices)):
        for j in range(i + 1, len(duplicate_indices)):
            if df.iloc[duplicate_indices[i]].equals(df.iloc[duplicate_indices[j]]):
                print(f"Ligne numéro {duplicate_indices[i]} en doublon avec ligne numéro {duplicate_indices[j]}")
                to_drop.append(duplicate_indices[j])

print("##################################################################################")

# Premier affichage des statistiques descriptives sur le dataframe
pd.set_option("display.max_columns", None)
print(df.describe())

print("##################################################################################")

# 2-Alimentation, nettoyage et traitement des données

# Suppression des lignes en doublons
print("\nLignes à supprimer (indices): ")
print(to_drop)

df_cleaned = df.drop(to_drop)

# Affichage des informations du nouveau dataframe
df_cleaned.info()

print("##################################################################################")

# Création d'une nouvelle colonne "season" à partir de la colonne "month"
# Définition des saisons
season_mapping = {
    "jan": "hiver", "feb": "hiver", "dec": "hiver",
    "mar": "printemps", "apr": "printemps", "may": "printemps",
    "jun": "ete", "jul": "ete", "aug": "ete",
    "sep": "automne", "oct": "automne", "nov": "automne"
}

df_cleaned["season"] = df_cleaned["month"].map(season_mapping).astype("category")
df_cleaned.info()

print("##################################################################################")

# Ajout d'une colonne calculant l'indice BUI pour en déduire l'indice FWI à partir de la source https://wikifire.wsl.ch/tiki-index8720.html?page=Buildup+index
def calculate_bui(dmc, dc):
    condition = dmc <= 0.4 * dc
    bui = np.where(
        condition,
        (0.8 * dmc * dc) / (dmc + 0.4 * dc),
        dmc - (1 - (0.8 * dc) / (dmc + 0.4 * dc)) *
        (0.92 + (0.0114 * dmc) ** 1.7)
    )
    rounded_bui = np.round(bui, 1)
    return rounded_bui

df_cleaned["BUI"] = calculate_bui(df_cleaned["DMC"], df_cleaned["DC"])

# Caster la colonne "BUI" en type "float"
df_cleaned["BUI"] = df_cleaned["BUI"].astype("float")

# Vérifier le type de la colonne
print(df_cleaned["BUI"].dtype)

# Affichage du DataFrame avec la nouvelle colonne
print("Premieres lignes du dataframe avec la colonne BUI: ")
print(df_cleaned.head())

# Ajout d'une colonne calculant l'indice FWI à partir d'une formule simplifiée selon publication de Van Wagner (1987)
def calculate_fwi(isi, bui):
    fwi = np.sqrt(0.1 * isi * bui)
    rounded_fwi = np.round(fwi, 1)
    return rounded_fwi

df_cleaned["FWI"] = calculate_fwi(df_cleaned["ISI"], df_cleaned["BUI"])

# Caster la colonne "FWI" en type "float"
df_cleaned["FWI"] = df_cleaned["FWI"].astype("float")

# Vérifier le type de la colonne
print(df_cleaned["FWI"].dtype)

# Affichage de la colonne "FWI"
print("Affichage de la colonne FWI: ")

# Désactiver la limitation d'affichage
pd.set_option("display.max_rows", None)

print(df_cleaned["FWI"])

# Afficher la valeur max de la colonne "FWI"
print(f"La valeur maximale de l'indice FWI dans le dataframe est: {df_cleaned['FWI'].max()}.")

# Ajout d'une colonne "bscale" mesurant l'accumulation de combustible
def get_bscale(bui):
    if bui < 19:
        return "b1"
    elif 19 <= bui < 39:
        return "b2"
    elif 39 <= bui < 59:
        return "b3"
    else:
        return "b4"

# Ajout d'une colonne "risque_lie_au_combustible"
def get_bscale_description(bui):
    if bui < 19:
        return "faible accumulation de combustible"
    elif 19 <= bui < 39:
        return "accumulation moderee"
    elif 39 <= bui < 59:
        return "accumulation elevee"
    else:
        return "accumulation extreme"

# Ajout d'une colonne "iscale" mesurant la rapidité de propagation du feu
def get_iscale(isi):
    if isi < 3:
        return "i1"
    elif 3 <= isi < 7:
        return "i2"
    elif 7 <= isi < 12:
        return "i3"
    else:
        return "i4"

# Ajout d'une colonne "risque_lie_a_propagation"
def get_iscale_description(isi):
    if isi < 3:
        return "propagation tres lente"
    elif 3 <= isi < 7:
        return "propagation moderee"
    elif 7 <= isi < 12:
        return "propagation rapide"
    else:
        return "propagation extreme"

# Ajouter d'une colonne "niveau de danger" selon source https://forest-fire.emergency.copernicus.eu/about-effis/technical-background/fire-danger-forecast
def get_danger_level(fwi):
    if fwi < 2:  # 5.2
        return "conditions humides, feu peu probable"
    elif 2 <= fwi < 5:  # 11.2
        return "risque modéré, feu possible sous certaines conditions"
    elif 5 <= fwi < 8:  # 21.3
        return "risque accru, feu rapide en présence de vent"
    elif 8 <= fwi < 11:  # 38
        return "propagation rapide, feux difficiles à controler"
    else:  # 50
        return "conditions critiques, incendies intenses et incontrolables"

# Ajouter d'une colonne "description du niveau"
def get_level_description(fwi):
    if fwi < 5.2:
        return "Peu ou pas de risque d'incendie"
    elif 5.2 <= fwi < 11.2:
        return "Risque d'incendie faible, contrôle possible"
    elif 11.2 <= fwi < 21.3:
        return "Risque d'incendie modéré, nécessite une attention accrue"
    elif 21.3 <= fwi < 38.0:
        return "Risque important, incendies se propagent rapidement"
    elif 38.0 <= fwi < 50.0:
        return "Conditions très sèches, risque de propagation rapide"
    else:
        return "Conditions extrêmes, très grand risque d'incendie"

# Ajouter les six nouvelles colonnes au dataframe
df_cleaned["bscale"] = df_cleaned["BUI"].apply(get_bscale)
df_cleaned["bscale_description"] = df_cleaned["BUI"].apply(get_bscale_description)
df_cleaned["iscale"] = df_cleaned["ISI"].apply(get_iscale)
df_cleaned["iscale_description"] = df_cleaned["ISI"].apply(get_iscale_description)
df_cleaned["danger_level"] = df_cleaned["FWI"].apply(get_danger_level)
df_cleaned["level_description"] = df_cleaned["FWI"].apply(get_level_description)

# Caster les six nouvelles colonnes en type "category"
df_cleaned["bscale"] = df_cleaned["bscale"].astype("category")
df_cleaned["bscale_description"] = df_cleaned["bscale_description"].astype("category")
df_cleaned["iscale"] = df_cleaned["iscale"].astype("category")
df_cleaned["iscale_description"] = df_cleaned["iscale_description"].astype("category")
df_cleaned["danger_level"] = df_cleaned["danger_level"].astype("category")
df_cleaned["level_description"] = df_cleaned["level_description"].astype("category")

# Vérifier le type des nouvelles colonnes
print(df_cleaned["bscale"].dtype)
print(df_cleaned["bscale_description"].dtype)
print(df_cleaned["iscale"].dtype)
print(df_cleaned["iscale_description"].dtype)
print(df_cleaned['danger_level'].dtype)
print(df_cleaned['level_description'].dtype)

# Affichage des six nouvelles colonnes du DataFrame

# Désactiver la limitation d'affichage
pd.set_option("display.max_rows", None)

print(df_cleaned[["bscale", "bscale_description", "iscale", "iscale_description", "danger_level", "level_description"]])

print("##################################################################################")

# Comptage des lignes où la colonne "area" (surface brûlée) est égale à 0 et est différente de 0
count_area_0 = (df_cleaned["area"] == 0).sum()  # Nombre de lignes avec area == 0
count_area_non_0 = (df_cleaned["area"] != 0).sum()  # Nombre de lignes avec area != 0

# Nombre total de lignes du dataframe
nombre_lignes = df_cleaned.shape[0]

print(f"Le dataframe a un nombre total de lignes de: {nombre_lignes} dont {count_area_0} avec surface brûlée nulle.")

print("##################################################################################")

# Ajout d'une colonne log(1 + "area") pour réduire l'asymétrie de la distribution des valeurs de la colonne area
df_cleaned["log_area"] = np.log1p(df_cleaned["area"])

print("##################################################################################")

# Création de deux DataFrames sous-ensembles de df_cleaned (area = 0 et area != 0)

# Filtrer les lignes où la colonne "area" est égale à 0
df_area_0 = df_cleaned[df_cleaned["area"] == 0]

# Filtrer les lignes où la colonne "area" est différente de 0
df_area_non_0 = df_cleaned[df_cleaned["area"] != 0]

# Afficher les DataFrames résultants
print("DataFrame avec area = 0: ")
print(df_area_0.head())

print("\nDataFrame avec area != 0: ")
print(df_area_non_0.head())

print("##################################################################################")

# 3-Analyse et Visualisation des données

# Analyses univariées

# Création du répertoire pour les analyses univariées
save_dir = "1_viken_m2icdsd_2025_b2_analyse_distrib_univariee"
os.makedirs(save_dir, exist_ok=True)

# Mesure de l'aplatissement de chaque distribution (kurtosis)
# Création d'un fichier PDF pour sauvegarder l'analyse
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Titre du PDF
pdf.set_font("Arial", 'B', 16)
pdf.cell(200, 10, txt="Analyse de la Kurtosis et de la Distribution", ln=True, align="C")

# Liste des colonnes à tester
columns_to_test = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "BUI", "temp", "RH", "wind", "rain", "area", "log_area", "FWI"]

# Tester la kurtosis pour chaque colonne et écrire dans le PDF
pdf.ln(10)  # espace après le titre

# Mettre en gras pour l'en-tête des colonnes
pdf.set_font("Arial", 'B', 12)
pdf.cell(60, 10, txt="Colonne", border=1, align="C")
pdf.cell(40, 10, txt="Kurtosis", border=1, align="C")
pdf.cell(90, 10, txt="Type de Distribution", border=1, align="C")
pdf.ln()

# Retour à la police normale pour les résultats
pdf.set_font("Arial", '', 12)

for col in columns_to_test:
    kurt = df_cleaned[col].kurtosis()

    # Déterminer si la distribution est proche de normale (kurtosis proche de 3)
    if 2.5 <= kurt <= 3.5:
        dist_type = "Distribution proche de normale"
    elif kurt > 3:
        dist_type = "Distribution leptokurtique (queues lourdes)"
    else:
        dist_type = "Distribution platykurtique (queues légères)"

    # Afficher le résultat
    print(f"Colonne: {col}")
    print(f"Kurtosis: {kurt}")
    print(f"Type de distribution: {dist_type}")
    print("-" * 40)

    # Afficher chaque ligne avec les résultats
    pdf.cell(60, 10, txt=col, border=1, align="C")
    pdf.cell(40, 10, txt=str(round(kurt, 2)), border=1, align="C")
    pdf.cell(90, 10, txt=dist_type, border=1, align="C")
    pdf.ln()

# Sauvegarder le PDF dans le répertoire spécifié
pdf_output_path = os.path.join(save_dir, "analyse_kurtosis_distrib.pdf")
pdf.output(pdf_output_path)

print(f"Le fichier PDF a été créé : {pdf_output_path}")

# Mesure de la distribution des données de chaque colonne par rapport à une distribution normale (skewness)

# Création d'un fichier PDF pour sauvegarder l'analyse
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Titre du PDF
pdf.set_font("Arial", 'B', 16)
pdf.cell(200, 10, txt="Analyse de la Skewness et de la Distribution", ln=True, align="C")

# Liste des colonnes à tester
columns_to_test = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "BUI", "temp", "RH", "wind", "rain", "area", "log_area", "FWI"]

# Tester la skewness pour chaque colonne et écrire dans le PDF
pdf.ln(10)  # espace après le titre

# Mettre en gras pour l'en-tête des colonnes
pdf.set_font("Arial", 'B', 12)
pdf.cell(60, 10, txt="Colonne", border=1, align="C")
pdf.cell(40, 10, txt="Skewness", border=1, align="C")
pdf.cell(90, 10, txt="Type de Distribution", border=1, align="C")
pdf.ln()

# Retour à la police normale pour les résultats
pdf.set_font("Arial", '', 12)

for col in columns_to_test:
    skew = df_cleaned[col].skew()

    # Déterminer si la distribution est proche de normale (skewness proche de 0)
    if -0.5 <= skew <= 0.5:
        dist_type = "Distribution proche de normale"
    elif skew > 0:
        dist_type = "Distribution asymétrique à droite"
    else:
        dist_type = "Distribution asymétrique à gauche"

    # Afficher chaque ligne avec les résultats
    pdf.cell(60, 10, txt=col, border=1, align="C")
    pdf.cell(40, 10, txt=str(round(skew, 2)), border=1, align="C")
    pdf.cell(90, 10, txt=dist_type, border=1, align="C")
    pdf.ln()

# Sauvegarder le PDF dans le répertoire spécifié
pdf_output_path = os.path.join(save_dir, "analyse_skewness_distrib.pdf")
pdf.output(pdf_output_path)

print(f"Le fichier PDF a été créé : {pdf_output_path}")

print("##################################################################################")

# Visualisation de la distribution des valeurs de chaque colonne

# Configuration du style des graphes
sns.set_style("whitegrid")

# Séparation des variables numériques et celles de type "category"
num_vars = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "BUI", "temp", "RH", "wind", "rain", "area", "log_area", "FWI"]
cat_vars = ["month", "day", "season", "bscale", "bscale_description", "iscale", "iscale_description", "danger_level", "level_description"]

# Visualisation des variables numériques
for col in num_vars:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_cleaned[col], bins=10, kde=True, color="royalblue")
    plt.title(f"Distribution de {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Fréquence", fontsize=12)

    # Sauvegarde des figures en PNG
    plt.savefig(os.path.join(save_dir, f"{col}.png"), format="png", dpi=300)
    plt.close()

# Visualisation des variables catégoriques
for col in cat_vars:
    # Trier les catégories par ordre croissant
    sorted_categories = sorted(df_cleaned[col].unique())
    plt.figure(figsize=(45, 30))
    sns.countplot(x=df_cleaned[col], palette="viridis", order=sorted_categories)
    plt.title(f"Répartition de {col}", fontsize=40)
    plt.xlabel(col, fontsize=22)
    plt.ylabel("Nombre d'observations", fontsize=35)
    plt.xticks(rotation=10, fontsize = 35, fontweight ="bold")

    # Taille des ticks de l'axe y (valeurs sur l'axe des ordonnées)
    plt.yticks(fontsize=35, fontweight="bold")

    # Sauvegarde des figures en PNG
    plt.savefig(os.path.join(save_dir, f"{col}.png"), format="png", dpi=300)
    plt.close()

print(f"Les graphiques de distribution de chaque colonne sont enregistrés dans le dossier : {save_dir}")

# Affichage de plusieurs courbes sur une même page

# Configuration du style des graphes
sns.set_style("whitegrid")

# Graphiques X et Y sur la même figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

sns.countplot(x=df_cleaned["X"], palette="viridis", ax=axes[0])
axes[0].set_title("Répartition de X", fontsize=14)
axes[0].set_xlabel("X", fontsize=12)
axes[0].set_ylabel("Nombre d'observations", fontsize=12)

sns.countplot(x=df_cleaned["Y"], palette="viridis", ax=axes[1])
axes[1].set_title("Répartition de Y", fontsize=14)
axes[1].set_xlabel("Y", fontsize=12)
axes[1].set_ylabel("Nombre d'observations", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "X_Y.png"), format="png", dpi=300)
plt.close()

# Graphiques FFMC, DMC, DC, ISI, BUI, FWI sur la même figure
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 15))
fire_vars = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]

for i, col in enumerate(fire_vars):
    sns.histplot(df_cleaned[col], bins=10, kde=True, ax=axes[i // 2, i % 2], color="royalblue")
    axes[i // 2, i % 2].set_title(f"Distribution de {col}", fontsize=14)
    axes[i // 2, i % 2].set_xlabel(col, fontsize=12)
    axes[i // 2, i % 2].set_ylabel("Fréquence", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "FFMC_DMC_DC_ISI_BUI_FWI.png"), format="png", dpi=300)
plt.close()

# Graphiques temp, RH, wind, rain sur la même figure
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
weather_vars = ["temp", "RH", "wind", "rain"]

for i, col in enumerate(weather_vars):
    sns.histplot(df_cleaned[col], bins=10, kde=True, ax=axes[i // 2, i % 2], color="darkorange")
    axes[i // 2, i % 2].set_title(f"Distribution de {col}", fontsize=14)
    axes[i // 2, i % 2].set_xlabel(col, fontsize=12)
    axes[i // 2, i % 2].set_ylabel("Fréquence", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "temp_RH_wind_rain.png"), format="png", dpi=300)
plt.close()

print(f"Les ensembles de graphiques donnant la distribution des colonnes sont enregistrés dans le dossier : {save_dir}")

# Diagramme en barres pour comptage des valeurs où "area" = 0 et "area" > 0
df_count = pd.DataFrame({
    "Condition": ["Surface Brûlée = 0", "Surface Brûlée > 0"],
    "Count": [sum(df_cleaned["area"] == 0), sum(df_cleaned["area"] > 0)]
})

plt.figure(figsize=(7, 5))
sns.barplot(x='Condition', y='Count', data=df_count, palette='Blues')

plt.title("Nombre de lignes avec area égal à 0 et area différent de 0", fontsize=14, fontweight='bold')
plt.xlabel("Condition", fontsize=12)
plt.ylabel("Nombre de lignes", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Sauvegarde du diagramme en barres
file_path_png = os.path.join(save_dir, "area_distribution.png")
plt.savefig(file_path_png, format="png", dpi=300)
plt.close()

# Histogrammes des colonnes "area" et "log_area" pour visualiser leur distribution
# Créer un document avec deux graphiques (2 lignes, 1 colonne)
plt.figure(figsize=(12, 8))

# Premier histogramme : Distribution de "area"
plt.subplot(2, 1, 1)  # 2 lignes, 1 colonne, position 1
sns.histplot(df_cleaned["area"], bins=30, kde=False, color='skyblue', edgecolor='black')

# Titre et labels du premier graphique
plt.title('Répartition des valeurs de la colonne "area"', fontsize=14, fontweight='bold')
plt.xlabel('Surface brûlée area en hectare', fontsize=12)
plt.ylabel('Fréquence', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Deuxième histogramme : Distribution de 'log1p(area)'
plt.subplot(2, 1, 2)  # 2 lignes, 1 colonne, position 2
sns.histplot(df_cleaned["log_area"], bins=30, kde=False, color='lightgreen', edgecolor='black')

# Titre et labels du deuxième graphique
plt.title('Répartition des valeurs de "log_area"', fontsize=14, fontweight='bold')
plt.xlabel('log1p(Surface)', fontsize=12)
plt.ylabel('Fréquence', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ajuster l'espacement entre les graphiques
plt.tight_layout()

# Sauvegarder l'image dans un fichier
file_path_png = os.path.join(save_dir, "area_and_log_area_histograms_distribution.png")
plt.savefig(file_path_png, format="png", dpi=300)

# Fermer la figure sans l'afficher
plt.close()

print("##################################################################################")

# Statistiques descriptives

# Création du répertoire pour sauvegarder les fichiers
save_dir = "2_viken_m2icdsd_2025_b2_desc_stats"
os.makedirs(save_dir, exist_ok=True)

# Liste des regroupements de variables pour les pages du PDF
plots_groups = {
    "X_Y": ["X", "Y"],
    "FFMC_DMC_DC_ISI_BUI_FWI": ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"],
    "Temp_RH_Wind_Rain": ["temp", "RH", "wind", "rain"],
    "Area": ["area", "log_area"]
}

# Génération des boxplots
for page_name, cols in plots_groups.items():
    fig, axes = plt.subplots(nrows=1, ncols=len(cols), figsize=(6 * len(cols), 6))

    # S'assurer que axes est toujours une liste
    if len(cols) == 1:
        axes = [axes]  # Convertir en liste si une seule variable

    # Création des boxplots
    for i, col in enumerate(cols):
        ax = axes[i]
        sns.boxplot(y=df_cleaned[col], ax=ax, color="royalblue", width=0.5)

        # Ajout du titre et labels
        ax.set_title(f"Boxplot de {col}", fontsize=14, fontweight='bold')
        ax.set_ylabel(col, fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Sauvegarde en PNG
    png_path = os.path.join(save_dir, f"boxplot_{page_name}.png")
    plt.savefig(png_path, dpi=300)

    plt.close()

print(f"Boxplots pour statistiques descriptives sauvegardés dans {save_dir}.")

print("##################################################################################")

# Calcul des statistiques descriptives avec Numpy
statistics = {}

for col in df_cleaned.select_dtypes(include=[np.number]).columns:
    # Calcul des quartiles et de l'IQR
    Q1 = np.percentile(df_cleaned[col], 25)
    Q3 = np.percentile(df_cleaned[col], 75)
    IQR = Q3 - Q1

    # Calcul de la limite des moustaches supérieures
    whisker_upper = Q3 + 1.5 * IQR

    # Vérification si Q4 (maximum) est un outlier
    max_value = np.max(df_cleaned[col])
    is_outlier = max_value > whisker_upper

    statist = {
        "Moyenne": np.mean(df_cleaned[col]),
        "Médiane": np.median(df_cleaned[col]),
        "Variance": np.var(df_cleaned[col]),
        "Écart type": np.std(df_cleaned[col]),
        "Minimum": np.min(df_cleaned[col]),
        "Q1 (25%)": Q1,
        "Q3 (75%)": Q3,
        "IQR": IQR,
        "Maximum": max_value,
        "Outlier (Q4)": "Oui" if is_outlier else "Non",
    }

    # Arrondir toutes les valeurs numériques à 2 décimales
    statist = {k: (round(v, 2) if isinstance(v, (int, float)) else v) for k, v in statist.items()}

    statistics[col] = statist

# Création du DataFrame des statistiques
desc_stats_df = pd.DataFrame(statistics).T

# Création du fichier PDF pour sauvegarder
pdf_path = os.path.join(save_dir, "descriptive_statistics.pdf")

with PdfPages(pdf_path) as pdf:
    # Tracer les statistiques sous forme de tableau
    fig, ax = plt.subplots(figsize=(12, 8))  # Taille du graphique pour le tableau
    ax.axis('off')
    ax.table(cellText=desc_stats_df.values,
             colLabels=desc_stats_df.columns,
             rowLabels=desc_stats_df.index,
             loc='center',
             cellLoc='center',
             colLoc='center',
             bbox=[0, 0, 1, 1],
             fontsize=14)

    # Sauvegarder le tableau en PDF
    pdf.savefig(fig)

    plt.close()

print(f"Statistiques descriptives sauvegardées dans {save_dir}.")

print("##################################################################################")

# Analyses bivariées

# Définir un répertoire et le créer si non encore existant
save_dir = "3_viken_m2icdsd_2025_b2_analyses_bivariées"
os.makedirs(save_dir, exist_ok=True)

# Visualisation de la distribution mensuelle de la surface brûlée sur le dataframe avec surface brûlée non nulle

# Liste des mois dans l'ordre du calendrier
mois_ordre = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]

# Boxplot de la distribution de la surface brûlée par mois
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_area_non_0, x='month', y='area', palette="Set2", order=mois_ordre)
plt.title("Distribution mensuelle de la surface brûlée (ha)")
plt.xlabel("Mois")
plt.ylabel("Surface brûlée area (ha)")
plt.xticks(rotation=45)
plt.tight_layout()

# Sauvegarde du graphique en .png
boxplot_filename_png = os.path.join(save_dir, "boxplot_surface_brulee_par_mois.png")
plt.savefig(boxplot_filename_png)
plt.close()

# Diagramme en barres pour la somme de la surface brûlée par mois
monthly_sum = df_area_non_0.groupby("month")["area"].sum().reset_index()  # Somme de la surface brûlée par mois
plt.figure(figsize=(10, 6))
sns.barplot(data=monthly_sum, x='month', y='area', palette="Set3", order=mois_ordre)
plt.title("Surface Brûlée Totale par Mois (ha)")
plt.xlabel("Mois")
plt.ylabel("Surface Brûlée Totale (ha)")
plt.xticks(rotation=45)
plt.tight_layout()

# Sauvegarde du graphique en .png
barplot_sum_filename_png = os.path.join(save_dir, "barplot_sum_surface_brulee.png")
plt.savefig(barplot_sum_filename_png)
plt.close()

# Diagramme en barres pour la moyenne de la surface brûlée par mois
monthly_avg = df_area_non_0.groupby("month")["area"].mean().reset_index()  # Moyenne de la surface brûlée par mois
plt.figure(figsize=(10, 6))
sns.barplot(data=monthly_avg, x='month', y='area', palette="Set3", order=mois_ordre)
plt.title("Surface Brûlée Moyenne par Mois (ha)")
plt.xlabel("Mois")
plt.ylabel("Surface Brûlée Moyenne (ha)")
plt.xticks(rotation=45)
plt.tight_layout()

# Sauvegarde du graphique en .png
barplot_avg_filename_png = os.path.join(save_dir, "barplot_avg_surface_brulee.png")
plt.savefig(barplot_avg_filename_png)
plt.close()

# Diagramme en barres pour la quantité maximale de pluie enregistrée par mois
monthly_max_rain = df_cleaned.groupby("month")["rain"].max().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=monthly_max_rain, x='month', y='rain', palette="Blues", order=mois_ordre)
plt.title("Quantité Maximale de Pluie Enregistrée par Mois")
plt.xlabel("Mois")
plt.ylabel("Quantité Maximale de Pluie (mm)")
plt.xticks(rotation=45)
plt.tight_layout()

# Sauvegarde du graphique en .png
barplot_max_rain_filename_png = os.path.join(save_dir, "barplot_max_rain.png")
plt.savefig(barplot_max_rain_filename_png)
plt.close()

# Diagramme en barre pour quantité maximale de pluie enregistrée par mois et afficher le jour correspondant
monthly_max_rain_day = df_cleaned.loc[df_cleaned.groupby("month")["rain"].idxmax(), ["month", "rain", "day"]].reset_index(drop=True)
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=monthly_max_rain_day, x='month', y='rain', palette="Blues", order=mois_ordre)

# Récupérer les positions réelles des barres sur l'axe X
x_positions = {month: i for i, month in enumerate(mois_ordre)}

# Ajout des labels du jour de la semaine au-dessus des barres (uniquement si rain > 0)
for _, row in monthly_max_rain_day.iterrows():
    if row["rain"] > 0:  # Vérifie que la quantité de pluie n'est pas nulle
        x_pos = x_positions[row["month"]]  # Trouver la vraie position X du mois
        plt.text(x_pos, row["rain"] + 0.2, row["day"], ha='center', fontsize=10, color='black', fontweight='bold')

plt.title("Quantité Maximale de Pluie par Mois et Jour Correspondant")
plt.xlabel("Mois")
plt.ylabel("Quantité Maximale de Pluie (mm)")
plt.xticks(rotation=45)
plt.tight_layout()

# Sauvegarde du graphique en .png
barplot_max_rain_png = os.path.join(save_dir, "barplot_max_rain_day.png")
plt.savefig(barplot_max_rain_png)
plt.close()

# Affichage des régions géographiques du parc qui connaissent les plus grands incendies
plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(
    data=df_cleaned,
    x="X", y="Y",
    size="area",  # La taille des cercles est proportionnelle à area
    sizes=(10, 500),  # Ajustement des tailles min et max des cercles
    hue="area",  # La couleur peut aussi varier en fonction de la surface brûlée
    palette="viridis",  # Palette de couleurs pour visualiser l'intensité
    edgecolor="black",  # Contour des cercles pour une meilleure visibilité
    alpha=0.6  # Transparence pour éviter un affichage trop chargé
)

# Titre et labels
plt.title("Régions du parc de Montesinho qui connaissent les plus grands incendies", fontsize=14)
plt.xlabel("X (Longitude)", fontsize=12)
plt.ylabel("Y (Latitude)", fontsize=12)
plt.legend(title="Surface Brûlée (ha)", loc="upper left", fontsize=10)

# Sauvegarde du graphique
scatterplot_filename = os.path.join(save_dir, "scatterplot_regionsparc_grands_incendies.png")
plt.savefig(scatterplot_filename)
plt.close()

print("Les graphiques de surface brûlée mensuelle ont été enregistrés.")

# Visualisation de la fréquence des incendies par mois
plt.figure(figsize=(10, 6))
sns.countplot(data=df_area_non_0, x='month', palette="Set2", order=mois_ordre)
plt.title("Fréquence des Incendies par Mois")
plt.xlabel("Mois")
plt.ylabel("Nombre d'Incendies")
plt.xticks(rotation=45)
plt.tight_layout()

# Sauvegarde du graphique
plt.savefig(f"{save_dir}/frequence_incendies_par_mois.png")
plt.close()

# Visualisation de la fréquence des incendies par saison
# Comptage du nombre d'incendies par saison
season_counts = df_area_non_0["season"].value_counts().reindex(["hiver", "printemps", "ete", "automne"])

# Création du diagramme en barres
plt.figure(figsize=(8, 6))
sns.barplot(x=season_counts.index, y=season_counts.values, palette="coolwarm")
plt.title("Fréquence des Incendies par Saison")
plt.xlabel("Saison")
plt.ylabel("Nombre d'Incendies")
plt.xticks(rotation=0)
plt.tight_layout()

# Sauvegarde du graphique en PNG
season_freq_png = os.path.join(save_dir, "frequence_incendies_saisons.png")
plt.savefig(season_freq_png)
plt.close()

print("Le graphique de la fréquence des incendies par saison a été enregistré.")

# Visualisation version 1 de la surface brûlée en fonction des coordonnées X et Y

# Créer une figure 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Définir les données
x = df_area_non_0["X"]  # Coordonnée X
y = df_area_non_0["Y"]  # Coordonnée Y
z = df_area_non_0["area"]  # Surface brûlée (area)

# Créer un scatter plot en 3D
ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', edgecolors='k', alpha=0.7)

# Ajouter les titres et les labels
ax.set_title('V1 Visualisation 3D de la Surface Brûlée (ha) en fonction de X et Y', fontsize=16)
ax.set_xlabel('Coordonnée X', fontsize=12)
ax.set_ylabel('Coordonnée Y', fontsize=12)
ax.set_zlabel('Surface Brûlée area (ha)', fontsize=12)

# Afficher la colorbar pour indiquer l'intensité des surfaces brûlées
cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Surface Brûlée (ha)', rotation=270, labelpad=20)

# Enregistrer le graphique en .png dans le répertoire
png_path = os.path.join(save_dir, 'visualisation_1_surface_brulee.png')
fig.savefig(png_path, format='png', bbox_inches='tight')

# Fermer la figure après enregistrement
plt.close()

# Message de confirmation
print(f"Le graphique a été enregistré dans le répertoire : {save_dir}")
print(f"Fichier .png enregistré sous : {png_path}")

# Version 2 avec surface de la visualisation de la surface brûlée en fonction des coordonnées X et Y
# Créez une figure 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Définir les données
x = df_area_non_0["X"]  # Coordonnée X
y = df_area_non_0["Y"]  # Coordonnée Y
z = df_area_non_0["area"]  # Surface brûlée (area)

# Créer une grille pour une meilleure représentation de la surface
X_grid, Y_grid = np.meshgrid(np.unique(x), np.unique(y))

# Interpolation des valeurs de surface pour obtenir des valeurs z sur la grille
Z_grid = np.zeros(X_grid.shape)
for i in range(X_grid.shape[0]):
    for j in range(X_grid.shape[1]):
        # Correspondance de X, Y avec les valeurs z dans le dataframe
        idx = (x == X_grid[i, j]) & (y == Y_grid[i, j])
        if np.any(idx):
            Z_grid[i, j] = z[idx].iloc[0]  # Prendre la valeur de la surface brûlée (area)

# Créer une surface lisse avec `plot_surface`
surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none', alpha=0.7)

# Ajouter les titres et les labels
ax.set_title('V2 Représentation de la Surface Brûlée area (ha) en fonction de X et Y', fontsize=16)
ax.set_xlabel('Coordonnée X', fontsize=12)
ax.set_ylabel('Coordonnée Y', fontsize=12)
ax.set_zlabel('Surface Brûlée area (ha)', fontsize=12)

# Afficher la colorbar pour indiquer l'intensité des surfaces brûlées
cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Surface Brûlée area (ha)', rotation=270, labelpad=20)

# Enregistrer le graphique en .png dans le répertoire
png_path = os.path.join(save_dir, 'visualisation_2_surface_brulee_3d.png')
fig.savefig(png_path, format='png', bbox_inches='tight')

# Fermer la figure après l'enregistrement
plt.close(fig)

# Afficher un message de confirmation
print(f"Le graphique a été enregistré dans le répertoire : {save_dir}")
print(f"Fichier .png enregistré sous : {png_path}")

# Version 3 avec interpolation linéaire pour obtenir surface de la visualisation de la surface brûlée en fonction des coordonnées X et Y
# Définir les données
x = df_area_non_0["X"].values  # Coordonnée X
y = df_area_non_0["Y"].values  # Coordonnée Y
z = df_area_non_0["area"].values  # Surface brûlée (area)

# Créer une grille régulière pour interpolation
grid_x, grid_y = np.meshgrid(
    np.linspace(x.min(), x.max(), 100),
    np.linspace(y.min(), y.max(), 100)
)

# Interpolation des valeurs de z (area) sur la grille
grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

# Créer une figure 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Tracer la surface interpolée
surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none', alpha=0.9)

# Ajouter les points réels sous forme de scatter plot
ax.scatter(x, y, z, c=z, cmap='inferno', edgecolors='k', s=20, alpha=1.0)

# Définir les limites pour Z
ax.set_zlim(z.min(), z.max())

# Ajouter les labels et le titre
ax.set_title('V3 Surface Brûlée area (ha) en fonction de X et Y', fontsize=16)
ax.set_xlabel('Coordonnée X', fontsize=12)
ax.set_ylabel('Coordonnée Y', fontsize=12)
ax.set_zlabel('Surface Brûlée area (ha)', fontsize=12)

# Ajouter une barre de couleur
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Surface Brûlée area (ha)', rotation=270, labelpad=20)

# Enregistrer le graphique
png_path = os.path.join(save_dir, 'surface_brulee_interpolee.png')
fig.savefig(png_path, format='png', bbox_inches='tight')

# Fermer la figure après enregistrement
plt.close()

# Message de confirmation
print(f"Graphique enregistré sous :\n{png_path}")

print("##################################################################################")

# Visualiser la surface brûlée en fonction de la température
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_area_non_0, x='temp', y='area', alpha=0.6, edgecolor=None, color="royalblue")
plt.title("Surface Brûlée en Fonction de la Température")
plt.xlabel("Température (°C)")
plt.ylabel("Surface Brûlée (ha)")
plt.grid(True)
plt.tight_layout()

# Sauvegarde du graphique
plt.savefig(f"{save_dir}/surface_brulee_vs_temperature.png")
plt.close()

# Visualistation avec axe secondaire log area
plt.figure(figsize=(10, 6))

# Axe principal : area vs temperature
ax1 = sns.scatterplot(data=df_area_non_0, x='temp', y='area',
                      alpha=0.6, edgecolor=None, color="royalblue", label="Surface Brûlée (ha)")

# Création d'un axe secondaire
ax2 = ax1.twinx()
sns.scatterplot(data=df_area_non_0, x='temp', y='log_area',
                alpha=0.6, edgecolor=None, color="darkorange", label="log(Surface Brûlée)")

# Titres et labels
ax1.set_xlabel("Temp (°C)")
ax1.set_ylabel("Surface Brûlée (ha)", color="royalblue")
ax2.set_ylabel("log(Surface Brûlée)", color="darkorange")

# Ajout d'une grille et d'un titre
plt.title("Surface Brûlée et log(Surface Brûlée) en Fonction de la Température")
ax1.grid(True, linestyle="--", alpha=0.5)

# Sauvegarde du graphique
plt.savefig(f"{save_dir}/surface_brulee_et_surfacelog_vs_temperature.png")
plt.close()

# Visualiser la surface brûlée en fonction de l'humidité relative
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_area_non_0, x='RH', y='area', alpha=0.6, edgecolor=None, color="royalblue")
plt.title("Surface Brûlée en Fonction de l'humidité relative")
plt.xlabel("RH (%)")
plt.ylabel("Surface Brûlée (ha)")
plt.grid(True)
plt.tight_layout()

# Sauvegarde du graphique
plt.savefig(f"{save_dir}/surface_brulee_vs_humidite_relative.png")
plt.close()

# Visualistation avec axe secondaire log area
plt.figure(figsize=(10, 6))

# Axe principal : area vs humidité relative
ax1 = sns.scatterplot(data=df_area_non_0, x='RH', y='area',
                      alpha=0.6, edgecolor=None, color="royalblue", label="Surface Brûlée (ha)")

# Création d'un axe secondaire
ax2 = ax1.twinx()
sns.scatterplot(data=df_area_non_0, x='RH', y='log_area',
                alpha=0.6, edgecolor=None, color="darkorange", label="log(Surface Brûlée)")

# Titres et labels
ax1.set_xlabel("HR (%)")
ax1.set_ylabel("Surface Brûlée (ha)", color="royalblue")
ax2.set_ylabel("log(Surface Brûlée)", color="darkorange")

# Ajout d'une grille et d'un titre
plt.title("Surface Brûlée et log(Surface Brûlée) en Fonction de l'humisité relative")
ax1.grid(True, linestyle="--", alpha=0.5)

# Sauvegarde du graphique
plt.savefig(f"{save_dir}/surface_brulee_et_surfacelog_vs_hr.png")
plt.close()

# Visualiser la surface brûlée en fonction du vent
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_area_non_0, x='wind', y='area', alpha=0.6, edgecolor=None, color="royalblue")
plt.title("Surface Brûlée en Fonction du vent")
plt.xlabel("vent (km/h)")
plt.ylabel("Surface Brûlée (ha)")
plt.grid(True)
plt.tight_layout()

# Sauvegarde du graphique
plt.savefig(f"{save_dir}/surface_brulee_vs_vent.png")
plt.close()

# Visualistation avec axe secondaire log area
plt.figure(figsize=(10, 6))

# Axe principal : area vs wind
ax1 = sns.scatterplot(data=df_area_non_0, x='wind', y='area',
                      alpha=0.6, edgecolor=None, color="royalblue", label="Surface Brûlée (ha)")

# Création d'un axe secondaire
ax2 = ax1.twinx()
sns.scatterplot(data=df_area_non_0, x='wind', y='log_area',
                alpha=0.6, edgecolor=None, color="darkorange", label="log(Surface Brûlée)")

# Titres et labels
ax1.set_xlabel("Vent (km/h)")
ax1.set_ylabel("Surface Brûlée (ha)", color="royalblue")
ax2.set_ylabel("log(Surface Brûlée)", color="darkorange")

# Ajout d'une grille et d'un titre
plt.title("Surface Brûlée et log(Surface Brûlée) en Fonction du Vent")
ax1.grid(True, linestyle="--", alpha=0.5)

# Sauvegarde du graphique
plt.savefig(f"{save_dir}/surface_brulee_et_surfacelog_vs_vent.png")
plt.close()

print("##################################################################################")

# Visualisation d'autres pairplots

# Fonction de sauvegarde pour les pairplots
def save_figure(fig, file_name, save_dir):
    fig.savefig(os.path.join(save_dir, f"{file_name}.png"), bbox_inches='tight')
    plt.close(fig)

# Liste des colonnes numériques à analyser
# Définition des groupes de colonnes
colonnes_normales = ["X", "Y", "BUI", "temp", "FWI"]
colonnes_asymetriques = ["FFMC", "DMC", "DC", "ISI", "RH", "wind", "rain"]

# Affichage du pairplot pour le cas où la surface brûlée est non nulle
pairplot_burned = sns.pairplot(df_area_non_0[colonnes_normales + colonnes_asymetriques + ["log_area"]])
plt.suptitle("Pairplot - Surface brûlée non nulle", y=1.02)
save_figure(pairplot_burned.fig, "pairplot_burned", save_dir)

# Affichage du pairplot pour le cas où la surface brûlée est nulle
pairplot_no_burn = sns.pairplot(df_area_0[colonnes_normales + colonnes_asymetriques])
plt.suptitle("Pairplot - Surface brûlée nulle", y=1.02)
save_figure(pairplot_burned.fig, "pairplot_no_burn", save_dir)

print("Les pairplots ont été générés et sauvegardés.")

# Tracé des valeurs min, max et moyennes du FWI en fonction du mois

df_grouped = df_cleaned.groupby("month")["FWI"].agg(["max", "min", "mean"])

# Remettre les mois dans l'ordre calendaire
df_grouped = df_grouped.sort_index()

# Création de la figure et des axes
plt.figure(figsize=(10, 6))

# Tracer les trois courbes
plt.plot(df_grouped.index, df_grouped["max"], label='FWI Max', color='red', marker='o')
plt.plot(df_grouped.index, df_grouped["min"], label='FWI Min', color='blue', marker='o')
plt.plot(df_grouped.index, df_grouped["mean"], label='FWI Moyen', color='green', marker='o')

# Ajouter des labels, un titre et une légende
plt.xlabel('Mois')
plt.ylabel('FWI')
plt.title('FWI Max, Min et Moyen en fonction du mois')

# Réordonner les mois de 1 à 12 sur l'axe x
mois_ordre = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
plt.xticks(ticks=df_grouped.index, labels=mois_ordre)

# Ajouter la légende
plt.legend()

# Sauvegarder le graphique sous .png
plt.savefig(f'{save_dir}/fwi_courbes_en_fonction_mois_calendaire.png')
plt.close()

print("##################################################################################")

# Analyses multivariées

# Définir un répertoire et le créer si non encore existant
save_dir = "4_viken_m2icdsd_2025_b2_analyses_multivariées"
os.makedirs(save_dir, exist_ok=True)

# Définition des groupes de colonnes
colonnes_normales = ["X", "Y", "BUI", "temp", "FWI"]
colonnes_asymetriques = ["FFMC", "DMC", "DC", "ISI", "RH", "wind", "rain"]

# Matrices de corrélation
correlation_pearson = df_cleaned[["log_area"]+colonnes_normales].corr(method='pearson')  # (Normales vs Normales)
correlation_spearman_asym = df_cleaned[["log_area"]+colonnes_asymetriques].corr(method='spearman')  # (Asymétriques vs Asymétriques)
correlation_spearman_mixed = df_cleaned[["log_area"]+colonnes_normales + colonnes_asymetriques].corr(method='spearman')  # (Tout en Spearman)

# Affichage des matrices
print("\n Matrice de Corrélation Pearson (Colonnes Normales):\n", correlation_pearson)
print("\n Matrice de Corrélation Spearman (Colonnes Asymétriques):\n", correlation_spearman_asym)
print("\n Matrice de Corrélation Spearman (Mélange Normales & Asymétriques):\n", correlation_spearman_mixed)

# Création des heatmaps
fig, axes = plt.subplots(1, 3, figsize=(25, 7))

# Heatmap Pearson (Normales vs Normales)
sns.heatmap(correlation_pearson, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[0])
axes[0].set_title(" Matrice de Corrélation de Pearson (Colonnes Normales)")

# Heatmap Spearman (Asymétriques vs Asymétriques)
sns.heatmap(correlation_spearman_asym, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[1])
axes[1].set_title(" Matrice de Corrélation de Spearman (Colonnes Asymétriques)")

# Heatmap Spearman (Mélange Normales et Asymétriques)
sns.heatmap(correlation_spearman_mixed, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[2])
axes[2].set_title(" Matrice de Corrélation de Spearman (Tout)")

# Sauvegarde des figures
plt.tight_layout()
plt.savefig(f"{save_dir}/correlation_heatmaps.png")
plt.close()

# Poursuite des analyses multivariées en distinguant les cas surface brûlée nulle et les cas surface brulée non nulle

# Définir un répertoire et le créer si non encore existant
save_dir = "5_viken_m2icdsd_2025_b2_analyses_multivariees_surface_brulee_0_et_non_0"
os.makedirs(save_dir, exist_ok=True)

# Statistiques descriptives pour les sous-groupes
print("Statistiques descriptives - Surface brûlée non nulle:")
print(df_area_non_0.describe())

print("\nStatistiques descriptives - Surface brûlée nulle:")
print(df_area_0.describe())

def save_heatmap(heatmap, file_name, save_dir):
    plt.savefig(os.path.join(save_dir, f"{file_name}.png"), bbox_inches='tight')
    plt.close()

# Liste des colonnes numériques à analyser
variables_numeriques = ["X", "Y", "BUI", "temp", "FFMC", "DMC", "DC", "ISI", "RH", "wind", "rain", "FWI"]

# Dictionnaire pour stocker les résultats de normalité
normality_results = {"brulé": {}, "non_brulé": {}}

# Vérification de la normalité pour chaque variable dans chaque sous-ensemble
for var in variables_numeriques:
    # Test de Shapiro-Wilk pour la normalité
    p_value_non_0 = stats.shapiro(df_area_non_0[var])[1] if len(df_area_non_0[var]) > 3 else 1
    p_value_0 = stats.shapiro(df_area_0[var])[1] if len(df_area_0[var]) > 3 else 1

    normality_results["brulé"][var] = p_value_non_0 > 0.05  # True si normal, False sinon
    normality_results["non_brulé"][var] = p_value_0 > 0.05

# Séparation dynamique des colonnes normales et asymétriques
colonnes_normales_brule = [var for var, is_normal in normality_results["brulé"].items() if is_normal]
colonnes_asymetriques_brule = [var for var, is_normal in normality_results["brulé"].items() if not is_normal]

colonnes_normales_non_brule = [var for var, is_normal in normality_results["non_brulé"].items() if is_normal]
colonnes_asymetriques_non_brule = [var for var, is_normal in normality_results["non_brulé"].items() if not is_normal]

# Corrélation Pearson pour les colonnes normales

if not colonnes_normales_brule:
    print("Aucune colonne normale pour la zone brûlée.")
else:
    print("Colonnes normales pour la zone brûlée :", colonnes_normales_brule)

if colonnes_normales_brule:
    pearson_corr_burned = df_area_non_0[["log_area"] + colonnes_normales_brule].corr(method='pearson')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pearson_corr_burned, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Heatmap Corrélation Pearson - Surface brûlée non nulle")
    save_heatmap(plt, "pearson_corr_burned", save_dir)

if colonnes_normales_non_brule:
    pearson_corr_no_burn = df_area_0[colonnes_normales_non_brule].corr(method='pearson')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pearson_corr_no_burn, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Heatmap Corrélation Pearson - Surface brûlée nulle")
    save_heatmap(plt, "pearson_corr_no_burn", save_dir)

# Corrélation Spearman pour les colonnes asymétriques
if colonnes_asymetriques_brule:
    spearman_corr_burned = df_area_non_0[["log_area"] + colonnes_asymetriques_brule].corr(method='spearman')
    plt.figure(figsize=(10, 6))
    sns.heatmap(spearman_corr_burned, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Heatmap Corrélation Spearman - Surface brûlée non nulle")
    save_heatmap(plt, "spearman_corr_burned", save_dir)

if colonnes_asymetriques_non_brule:
    spearman_corr_no_burn = df_area_0[colonnes_asymetriques_non_brule].corr(method='spearman')
    plt.figure(figsize=(10, 6))
    sns.heatmap(spearman_corr_no_burn, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Heatmap Corrélation Spearman - Surface brûlée nulle")
    save_heatmap(plt, "spearman_corr_no_burn", save_dir)

print("Les heatmaps ont été générées et sauvegardées.")

print("##################################################################################")

# Statistiques inférentielles pour étudier l'influence des colonnes sur la surface brûlée

# Définir un répertoire et le créer si non encore existant
save_dir = "6_viken_m2icdsd_2025_b2_statistiques_inférentielles"
os.makedirs(save_dir, exist_ok=True)

# Fonction pour sauvegarder les résultats et graphiques dans un PDF
def save_results_to_pdf(results, save_dir, file_name="results_stats_inferentielles_tmw.pdf"):
    # Créer un PDF
    pdf_path = os.path.join(save_dir, file_name)

    # Définir le document PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)

    # Définir les styles de texte pour le PDF
    styles = getSampleStyleSheet()
    style_normal = styles['Normal']
    style_heading = styles['Heading1']

    # Créer une liste pour les éléments du document (texte et images)
    story = []

    # Ajouter un titre à la première page
    title = Paragraph("Analyse des tests statistiques - Résultats", style_heading)
    story.append(title)
    story.append(Spacer(1, 12))  # Espacement après le titre

    # Ajouter les résultats de chaque test au PDF
    for result in results:
        # Ajouter chaque résultat au PDF sous forme de paragraphe
        paragraph = Paragraph(result, style_normal)
        story.append(paragraph)
        story.append(Spacer(1, 12))  # Espacement entre les paragraphes

    # Sauvegarder le PDF
    doc.build(story)

# Variables à tester
variables = ["X", "Y", "BUI", "temp", "FFMC", "DMC", "DC", "ISI", "RH", "wind", "FWI"]

# Pour stocker les résultats
test_results = []

# Analyser la distribution et appliquer les tests
for var in variables:
    # Histogramme de la distribution pour les deux groupes
    plt.figure(figsize=(8, 5))
    sns.histplot(df_area_non_0[var], kde=True, color='blue', label='Surface brûlée non nulle')
    sns.histplot(df_area_0[var], kde=True, color='red', label='Surface brûlée nulle')
    plt.legend()
    plt.title(f"Distribution de {var} pour chaque groupe")

    # Sauvegarder l'histogramme
    plt.savefig(os.path.join(save_dir, f"{var}_distribution.png"))
    plt.close()

    # Test de normalité pour chaque groupe
    _, p_burned = stats.shapiro(df_area_non_0[var])
    _, p_no_burn = stats.shapiro(df_area_0[var])

    test_result = f"Test de normalité pour {var}:\n  p-value (Surface brûlée non nulle) : {p_burned:.3f}\n  p-value (Surface brûlée nulle) : {p_no_burn:.3f}"
    test_results.append(test_result)

    # Si les deux groupes sont normalement distribués, utiliser le t-test
    if p_burned > 0.05 and p_no_burn > 0.05:
        t_stat, p_value = stats.ttest_ind(df_area_non_0[var], df_area_0[var])
        result = f"T-test pour {var} - Statistique t : {t_stat:.3f}, p-value : {p_value:.3f}"
        if p_value < 0.05:
            result += "\nIl y a une différence significative entre les groupes."
        else:
            result += "\nIl n'y a pas de différence significative entre les groupes."
        test_results.append(result)
    else:
        u_stat, p_value = stats.mannwhitneyu(df_area_non_0[var], df_area_0[var])
        result = f"Test de Mann-Whitney pour {var} - Statistique U : {u_stat:.3f}, p-value : {p_value:.3f}"
        if p_value < 0.05:
            result += "\nIl y a une différence significative entre les groupes."
        else:
            result += "\nIl n'y a pas de différence significative entre les groupes."
        test_results.append(result)

    test_results.append("-" * 50)

# Sauvegarder tous les résultats dans un fichier PDF
save_results_to_pdf(test_results, save_dir, file_name="results_stats_inferentielles_tmw.pdf")

print("Les résultats ont été sauvegardés dans un fichier PDF.")

# Test du Chi2

# Convertir X et Y en catégories pour le test du Chi²
df_area_non_0["X_cat"] = df_area_non_0["X"].astype("category")
df_area_non_0["Y_cat"] = df_area_non_0["Y"].astype("category")

# Effectuer les tests du Chi² et stocker les résultats
results = []

for col in ["month", "day", "X_cat", "Y_cat"]:
    contingency_table = pd.crosstab(df_area_non_0[col], df_area_non_0["area"] > 0)
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Interprétation du résultat
    if p < 0.05:
        interpretation = f"Il y a une relation significative entre {col} et la présence d'un incendie (p = {p:.4f})."
    else:
        interpretation = f"Aucune relation significative détectée entre {col} et la présence d'un incendie (p = {p:.4f})."

    # Stocker le résultat
    results.append((col, chi2, p, dof, interpretation))

# Création du PDF
pdf_path = os.path.join(save_dir, "chi2_results.pdf")

with PdfPages(pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Titre du rapport
    text = "Test du Chi² : Analyse des relations entre variables catégoriques et présence d'incendie\n\n"
    text += "\n".join([f"{col}: {interp}" for col, _, p, _, interp in results])

    ax.text(0, 1, text, fontsize=12, va="top")

    pdf.savefig(fig)
    plt.close()

print(f"Rapport du test du Chi² sauvegardé dans {pdf_path}")

# ANOVA et test de Kruskal-Wallis pour comparer les surfaces brulées selon catégories des différentes colonnes

# Liste des variables continues (excluant area et log_area)
variables_continues = ["X", "Y", "BUI", "temp", "FFMC", "DMC", "DC", "ISI", "RH", "wind", "FWI"]

# Dictionnaire pour stocker les résultats
test_results = []

# Catégorisation des variables continues en classes
df_area_0_cat = df_area_0.copy()
df_area_non_0_cat = df_area_non_0.copy()

for var in variables_continues:
    df_area_0_cat[var + "_cat"] = pd.qcut(df_area_0[var], q=3, labels=["Bas", "Moyen", "Élevé"])
    df_area_non_0_cat[var + "_cat"] = pd.qcut(df_area_non_0[var], q=3, labels=["Bas", "Moyen", "Élevé"])

# Séparation des variables normales et asymétriques via le test de Shapiro-Wilk
variables_normales = []
variables_asymetriques = []

for var in variables_continues:
    p_value_non_0 = stats.shapiro(df_area_non_0[var])[1] if len(df_area_non_0[var]) > 3 else 1
    p_value_0 = stats.shapiro(df_area_0[var])[1] if len(df_area_0[var]) > 3 else 1

    if p_value_non_0 > 0.05 and p_value_0 > 0.05:
        variables_normales.append(var)
    else:
        variables_asymetriques.append(var)

# ANOVA pour les variables normales
for var in variables_normales:
    f_stat, p_value = stats.f_oneway(df_area_non_0[var], df_area_0[var])
    result = f"ANOVA pour {var} : F = {f_stat:.3f}, p-value = {p_value:.3f}"
    result += "\n-> Différence significative entre les groupes." if p_value < 0.05 else "\n-> Aucune différence significative."
    test_results.append(result)

# Kruskal-Wallis pour les variables asymétriques
for var in variables_asymetriques:
    h_stat, p_value = stats.kruskal(df_area_non_0[var], df_area_0[var])
    result = f"Kruskal-Wallis pour {var} : H = {h_stat:.3f}, p-value = {p_value:.3f}"
    result += "\n-> Différence significative entre les groupes." if p_value < 0.05 else "\n-> Aucune différence significative."
    test_results.append(result)

# Sauvegarde des résultats dans un PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.cell(200, 10, "Résultats des tests ANOVA et Kruskal-Wallis", ln=True, align='C')
pdf.ln(10)

for res in test_results:
    pdf.multi_cell(0, 10, res)
    pdf.ln(5)

pdf_file = os.path.join(save_dir, "anova_kruskal_results.pdf")
pdf.output(pdf_file)

print(f"Les résultats ont été sauvegardés dans {pdf_file}.")

print("##################################################################################")

# Analyses statistiques complémentaires avec R

# Définir un répertoire et le créer si non encore existant
save_dir = "7_viken_m2icdsd_2025_b2_statistiques_complementaires_avec_R"
os.makedirs(save_dir, exist_ok=True)

# Activer la conversion entre pandas et R
pandas2ri.activate()

# Importer les packages R nécessaires
base = importr('base')
stats = importr('stats')
ggplot2 = importr('ggplot2')
cluster = importr('cluster')
forecast = importr('forecast')

# Convertir le dataframe pandas en dataframe R
df_r = pandas2ri.py2rpy(df_cleaned)

# Test de normalité (Shapiro-Wilk et Kolmogorov-Smirnov) sur FWI
print("\nTest de normalité en R: ")
shapiro_test = stats.shapiro_test(df_r.rx2("FWI"))
ks_test = stats.ks_test(df_r.rx2("FWI"), "pnorm", mean=df_cleaned["FWI"].mean(), sd=df_cleaned["FWI"].std())
print(f"Shapiro-Wilk p-value: {shapiro_test[1]}")
print(f"Kolmogorov-Smirnov p-value: {ks_test[1]}")

# ANOVA (Analyse de la variance) pour tester les différences entre les saisons
print("\nAnalyse de la variance (ANOVA) en R entre FWI et season: ")
anova_model = stats.aov(r('FWI ~ season'), data=df_r)
print(base.summary(anova_model))
anova_summary = base.summary(anova_model)

# Ouvrir un fichier texte pour sauvegarder les résultats
with open("8_viken_m2icdsd_2025_b2_resultats_tests_R_normalite_FWI_ANOVA_FWI_season.txt", "w", encoding="utf-8") as file:
    # Écrire les résultats du test de normalité (Shapiro-Wilk et Kolmogorov-Smirnov)
    file.write("===== Test de normalité en R sur FWI =====\n")
    file.write(f"Shapiro-Wilk p-value: {shapiro_test[1]}\n")
    file.write(f"Kolmogorov-Smirnov p-value: {ks_test[1]}\n")

    # Ecrire les résultats de l'ANOVA
    file.write("\n===== Analyse de la variance (ANOVA) entre FWI et season =====\n")
    file.write(str(anova_summary))  # L'ANOVA génère un résumé qui peut être écrit directement en texte

print("Les résultats ont été sauvegardés.")

# 3. Régression non linéaire (polynomiale) entre FWI et Température
print("\nRégression polynomiale en R :")
poly_model = stats.lm(r('FWI ~ poly(temp, 2)'), data=df_r)
print(base.summary(poly_model))
poly_summary = base.summary(poly_model)

# 4. Régression linéaire multiple entre log_area et Température, Humidité relative (avec la surface brûlée non nulle)
df_r_area_non_0 = pandas2ri.py2rpy(df_area_non_0)

# temp et rh sont les variables explicatives
print("\nRégression linéaire multiple en R : Température et Humidité relative sur log surface brûlée non nulle")

# Créer le modèle de régression linéaire multiple
lm_model_rh_temp = r.lm('log_area ~ temp + RH', data=df_r_area_non_0)

# Résumé du modèle
print(base.summary(lm_model_rh_temp))
lm_summary_rh_temp = base.summary(lm_model_rh_temp)

# Régression linéaire multiple entre log_area et BUI, Température comme variables explicatives pour prédire la surface brûlée
print("\nRégression linéaire multiple en R : BUI et Température sur log surface brûlée non nulle")

# Création du modèle de régression linéaire multiple
lm_model_temp_bui = r.lm('log_area ~ temp + BUI', data=df_r_area_non_0)

# Résumé du modèle
print(base.summary(lm_model_temp_bui))
lm_summary_temp_bui = base.summary(lm_model_temp_bui)

# Sauvegarde des résultats dans un fichier texte
with open("9_viken_m2icdsd_2025_b2_regressions_polynom_multiple_results.txt", "w", encoding="utf-8") as file:
    # Régression polynomiale entre FWI et Température
    file.write("\n===== Régression Polynomiale entre FWI et Température =====\n")
    file.write(str(poly_summary))  # Résumé du modèle de régression polynomiale

    # Régression linéaire multiple entre log surface brûlée non nulle et Température, Humidité relative
    file.write(
        "\n===== Régression linéaire multiple entre log surface brûlée non nulle et Température, Humidité relative =====\n")
    file.write(str(lm_summary_rh_temp))  # Résumé du modèle de régression multiple avec Température et Humidité relative

    # Régression linéaire multiple entre log surface brûlée non nulle et BUI, Température pour prédire la surface brûlée
    file.write("\n===== Régression linéaire multiple entre log surface brûlée non nulle et BUI, Température=====\n")
    file.write(str(lm_summary_temp_bui))  # Résumé du modèle de régression multiple avec BUI et Température

# Test d'autres algorithmes

# Random Forest robuste à l'asymétrie des données

print("=================Essai Tests de modèles=======================")

# Régression avec random forest pour comprendre importance de wind et FFMC dans la prédiction de l'indice ISI
# Activer la conversion automatique Pandas -> R DataFrame
pandas2ri.activate()

# Sélectionner uniquement les colonnes nécessaires
df_selected = df_area_non_0[["ISI", "wind", "FFMC"]]

# Vérifier les types de données
print(df_selected.dtypes)

# Convertir en DataFrame R et assigner dans l'environnement R
df_r = pandas2ri.py2rpy(df_selected)
ro.globalenv["df_r"] = df_r  # Assigner df_r dans l'environnement R

# Charger la bibliothèque randomForest en R
randomForest = importr("randomForest")

# Exécuter le Random Forest en R
rf_model = ro.r('''
    set.seed(123)  # Fixer le seed pour la reproductibilité
    rf_model <- randomForest(ISI ~ wind + FFMC, data=df_r, ntree=500, importance=TRUE)
    print(rf_model)
    rf_model
''')

importance = ro.r('importance(rf_model)')
print(importance)

# Convertir l'importance en dataframe R
importance_r_df = ro.r["data.frame"](importance)

# Convertir l'objet importance_r_df en un DataFrame pandas
importance_df = pandas2ri.rpy2py(importance_r_df)

# Sauvegarde des résultats dans un fichier texte
with open("10_viken_m2icdsd_2025_b2_ISI_tilde_wind_FFMC_random_forest_results.txt", "w", encoding="utf-8") as file:
    # Modèle Random Forest
    file.write("================= Modèle Random Forest =================\n")
    file.write(str(rf_model))  # Résumé du modèle Random Forest
    file.write("\n\n")

    # Importance des variables
    file.write("================= Importance des Variables =================\n")
    file.write(importance_df.to_string())  # Importance des variables dans le modèle Random Forest

# Afficher la courbe directement dans Python
ro.r('plot(rf_model)')

# Définir un fichier de sortie pour le graphique
file_path = "11_viken_m2icdsd_2025_b2_essairforestun.png"

# Sauvegarder la courbe dans un fichier PNG
ro.r(f'''
    png("{file_path}", width=800, height=600)
    plot(rf_model)  # Plot du modèle
    dev.off()
''')

print("===========================================")

# Entraînement d'un modèle random forest pour prédire variable cible ISI en fonction de wind et FFMC,
# mesure de leur contribution à la réduction de l'erreur de prédiction
# puis prédiction de l'ISI sur les données d'entraînement et visualisation des résultats

# Activer la conversion automatique Pandas -> R DataFrame
pandas2ri.activate()

# Sélectionner uniquement les colonnes nécessaires
df_selected = df_area_non_0[["ISI", "wind", "FFMC"]]

# Ajouter la colonne des vraies valeurs "ISI" dans le DataFrame
y_true = df_selected["ISI"]

# Convertir en DataFrame R et assigner dans l'environnement R
df_r = pandas2ri.py2rpy(df_selected)
ro.globalenv["df_r"] = df_r  # Assigner df_r dans l'environnement R
ro.globalenv["y_true"] = pandas2ri.py2rpy(y_true)  # Assigner y_true dans l'environnement R

# Charger la bibliothèque randomForest en R
randomForest = importr("randomForest")

# Exécuter le Random Forest en R
rf_model = ro.r('''
    set.seed(123)  # Fixer le seed pour la reproductibilité
    rf_model <- randomForest(ISI ~ wind + FFMC, data=df_r, ntree=500, importance=TRUE)
    print(rf_model)
    rf_model
''')

# Obtenir l'importance des variables
importance_r = ro.r('importance(rf_model)')
print(importance)

# Obtenir l'importance des variables sous forme de data.frame R
importance_r = ro.r('as.data.frame(importance(rf_model))')  # Convertir en data.frame

# Convertir l'importance (qui est un R object) en DataFrame pandas
importance_df = pandas2ri.rpy2py(importance_r)

# Obtenir les prédictions en R et les convertir en liste (non numpy)
y_pred_r = ro.r('predict(rf_model, df_r)')
y_pred_list = list(y_pred_r)  # Convertir en liste

# Convertir les prédictions en DataFrame pandas
y_pred_df = pd.DataFrame(y_pred_list, columns=["Predictions"])

# Obtenir les vraies valeurs (y_true) directement depuis df_r
y_true_r = ro.r('df_r$ISI')

# Créer un DataFrame pour les vraies valeurs (y_true) et les prédictions (y_pred)
y_true_df = pd.DataFrame(y_true)
y_true_df["Predictions"] = y_pred_df

# Tracer le graphique directement en R
ro.r('''
    # Convertir les valeurs en R pour plot
    y_true <- c({0})
    y_pred <- c({1})

    # Créer le graphique
    plot(y_true, y_pred, main="Comparaison entre isi_true et isi_pred en fonction de wind et ffmc avec random forest", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x pour la comparaison
'''.format(','.join(map(str, y_true_r)), ','.join(map(str, y_pred_r))))

# Sauvegarder le graphique dans un fichier PNG
file_path = "12_viken_m2icdsd_2025_b2_essai_random_forest_isi_prediction_vs_true_avec_ffmc_wind.png"

# Sauvegarder la courbe dans un fichier PNG
ro.r(f'''
    png("{file_path}", width=800, height=600)
    plot(y_true, y_pred, main="Comparaison entre isi_true et isi_pred en fonction de wind et ffmc", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x
    dev.off()
''')

# Sauvegarder les résultats dans un fichier texte
with open("13_viken_m2icdsd_2025_b2_essai_random_forest_results_et_pred_ISI_tilde_wind_FFMC.txt", "w", encoding="utf-8") as file:
    file.write("================= Résultats du modèle Random Forest ISI tilde wind, FFMC =================\n")
    file.write("\nImportance des variables :\n")
    file.write(importance_df.to_string())  # Écrire l'importance des variables dans le fichier
    file.write("\n\n===== Prédictions du modèle =====\n")
    file.write(y_true_df.to_string())  # Écrire les vraies valeurs et les prédictions dans le fichier

print("===========================================")

# Entraînement d'un model random forest pour prédire le FWI en fonction de temp, wind et rh ,
# mesurer l'impact de chaque variable sur la prédiction,
# puis comparer les prédictions du modèle avec les vraies valeurs du FWI à l'aide d'une courbe

# Activer la conversion automatique Pandas -> R DataFrame
pandas2ri.activate()

# Sélectionner uniquement les colonnes nécessaires
df_selected = df_area_non_0[["FWI", "temp", "wind", "RH"]]

# Ajouter la colonne des vraies valeurs "ISI" dans le DataFrame
y_true = df_selected["FWI"]

# Convertir en DataFrame R et assigner dans l'environnement R
df_r = pandas2ri.py2rpy(df_selected)
ro.globalenv["df_r"] = df_r  # Assigner df_r dans l'environnement R
ro.globalenv["y_true"] = pandas2ri.py2rpy(y_true)  # Assigner y_true dans l'environnement R

# Charger la bibliothèque randomForest en R
#randomForest = importr("randomForest")

# Exécuter le Random Forest en R
rf_model = ro.r('''
    set.seed(123)  # Fixer le seed pour la reproductibilité
    rf_model <- randomForest(FWI ~ temp + wind + RH, data=df_r, ntree=500, importance=TRUE)
    print(rf_model)
    rf_model
''')

# Obtenir l'importance des variables
importance_r = ro.r('importance(rf_model)')
print(importance)

# Obtenir l'importance des variables sous forme de data.frame R
importance_r = ro.r('as.data.frame(importance(rf_model))')  # Convertir en data.frame

# Convertir l'importance (qui est un R object) en DataFrame pandas
importance_df = pandas2ri.rpy2py(importance_r)

# Obtenir les prédictions en R et les convertir en liste (non numpy)
y_pred_r = ro.r('predict(rf_model, df_r)')
y_pred_list = list(y_pred_r)  # Convertir en liste

# Convertir les prédictions en DataFrame pandas
y_pred_df = pd.DataFrame(y_pred_list, columns=['Predictions'])

# Obtenir les vraies valeurs (y_true) directement depuis df_r
y_true_r = ro.r('df_r$FWI')

# Créer un DataFrame pour les vraies valeurs (y_true) et les prédictions (y_pred)
y_true_df = pd.DataFrame(y_true)
y_true_df["Predictions"] = y_pred_df

# Tracer le graphique directement en R
ro.r('''
    # Convertir les valeurs en R pour plot
    y_true <- c({0})
    y_pred <- c({1})

    # Créer le graphique
    plot(y_true, y_pred, main="Comparaison entre fwi_true et fwi_pred en fonction de temp, wind et rh avec random forest", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x pour la comparaison
'''.format(','.join(map(str, y_true_r)), ','.join(map(str, y_pred_r))))

# Sauvegarder le graphique dans un fichier PNG
file_path = "14_viken_m2icdsd_2025_b2_essai_random_forest_fwi_prediction_vs_true_avec_temp_wind_rh.png"

# Sauvegarder la courbe dans un fichier PNG
ro.r(f'''
    png("{file_path}", width=800, height=600)
    plot(y_true, y_pred, main="Comparaison entre fwi_true et fwi_pred en fonction de temp, winf et rh avec random forest", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x
    dev.off()
''')

# Sauvegarder les résultats dans un fichier texte
with open("15_viken_m2icdsd_2025_b2_essai_random_forest_results_et_pred_FWI_tilde_temp_wind_RH.txt", "w", encoding="utf-8") as file:
    file.write("================= Résultats du modèle Random Forest FWI tilde temp, wind, RH =================\n")
    file.write("\nImportance des variables :\n")
    file.write(importance_df.to_string())  # Écrire l'importance des variables dans le fichier
    file.write("\n\n===== Prédictions du modèle =====\n")
    file.write(y_true_df.to_string())  # Écrire les vraies valeurs et les prédictions dans le fichier

print("===========================================")

# Entraînement d'un modèle random forest pour prédire la surface brûlée area en fonction de "FFMC" et "ISI",
# détermination de la l'importance de chaque variable dans la prédiction de "area"
# puis comparer les prédictions du modèle avec les vraies valeurs de "area"

# Activer la conversion automatique Pandas -> R DataFrame
pandas2ri.activate()

# Sélectionner uniquement les colonnes nécessaires
df_selected = df_area_non_0[["area", "FFMC", "ISI"]]

# Ajouter la colonne des vraies valeurs "ISI" dans le DataFrame
y_true = df_selected["area"]

# Convertir en DataFrame R et assigner dans l'environnement R
df_r = pandas2ri.py2rpy(df_selected)
ro.globalenv["df_r"] = df_r  # Assigner df_r dans l'environnement R
ro.globalenv["y_true"] = pandas2ri.py2rpy(y_true)  # Assigner y_true dans l'environnement R

# Charger la bibliothèque randomForest en R
randomForest = importr("randomForest")

# Exécuter le Random Forest en R
rf_model = ro.r('''
    set.seed(123)  # Fixer le seed pour la reproductibilité
    rf_model <- randomForest(area ~ FFMC + ISI, data=df_r, ntree=500, importance=TRUE)
    print(rf_model)
    rf_model
''')

# Obtenir l'importance des variables
importance_r = ro.r('importance(rf_model)')
print(importance)

# Obtenir l'importance des variables sous forme de data.frame R
importance_r = ro.r('as.data.frame(importance(rf_model))')  # Convertir en data.frame

# Convertir l'importance (qui est un R object) en DataFrame pandas
importance_df = pandas2ri.rpy2py(importance_r)

# Obtenir les prédictions en R et les convertir en liste (non numpy)
y_pred_r = ro.r('predict(rf_model, df_r)')
y_pred_list = list(y_pred_r)  # Convertir en liste

# Convertir les prédictions en DataFrame pandas
y_pred_df = pd.DataFrame(y_pred_list, columns=["Predictions"])

# Obtenir les vraies valeurs (y_true) directement depuis df_r
y_true_r = ro.r('df_r$area')

# Créer un DataFrame pour les vraies valeurs (y_true) et les prédictions (y_pred)
y_true_df = pd.DataFrame(y_true)
y_true_df["Predictions"] = y_pred_df

# Tracer le graphique directement en R
ro.r('''
    # Convertir les valeurs en R pour plot
    y_true <- c({0})
    y_pred <- c({1})

    # Créer le graphique
    plot(y_true, y_pred, main="Comparaison entre area_true et area_pred en fonction de FFMC et ISI avec surface brûlée non nulle", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x pour la comparaison
'''.format(','.join(map(str, y_true_r)), ','.join(map(str, y_pred_r))))

# Sauvegarder le graphique dans un fichier PNG
file_path = "16_viken_m2icdsd_2025_b2_essai_random_forest_area_tilde_FFMC_ISI_prediction_vs_true_area_non_nulle.png"

# Sauvegarder la courbe dans un fichier PNG
ro.r(f'''
    png("{file_path}", width=800, height=600)
    plot(y_true, y_pred, main="Comparaison entre area_true et area_pred en fonction de FFMC et ISI area non nulle", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x
    dev.off()
''')

# Sauvegarder les résultats dans un fichier texte
with open("17_viken_m2icdsd_2025_b2_essai_random_forest_results_et_pred_area_tilde_FFMC_ISI_area_non_nulle.txt", "w", encoding="utf-8") as file:
    file.write("================= Résultats du modèle Random Forest area tilde FFMC, ISI area non nulle =================\n")
    file.write("\nImportance des variables :\n")
    file.write(importance_df.to_string())  # Écrire l'importance des variables dans le fichier
    file.write("\n\n===== Prédictions du modèle =====\n")
    file.write(y_true_df.to_string())  # Écrire les vraies valeurs et les prédictions dans le fichier

print("===========================================")

# Entraînement d'un modèle random forest pour prédire la surface brûlée area en fonction de "DMC" et "DC",
# détermination de la l'importance de chaque variable dans la prédiction de "area"
# puis comparer les prédictions du modèle avec les vraies valeurs de "area"

# Activer la conversion automatique Pandas -> R DataFrame
pandas2ri.activate()

# Sélectionner uniquement les colonnes nécessaires
df_selected = df_area_non_0[["area", "DMC", "DC"]]

# Ajouter la colonne des vraies valeurs 'ISI' dans le DataFrame
y_true = df_selected["area"]

# Convertir en DataFrame R et assigner dans l'environnement R
df_r = pandas2ri.py2rpy(df_selected)
ro.globalenv["df_r"] = df_r  # Assigner df_r dans l'environnement R
ro.globalenv["y_true"] = pandas2ri.py2rpy(y_true)  # Assigner y_true dans l'environnement R

# Charger la bibliothèque randomForest en R
randomForest = importr("randomForest")

# Exécuter le Random Forest en R
rf_model = ro.r('''
    set.seed(123)  # Fixer le seed pour la reproductibilité
    rf_model <- randomForest(area ~ DMC + DC, data=df_r, ntree=500, importance=TRUE)
    print(rf_model)
    rf_model
''')

# Obtenir l'importance des variables
importance_r = ro.r('importance(rf_model)')
print(importance)

# Obtenir l'importance des variables sous forme de data.frame R
importance_r = ro.r('as.data.frame(importance(rf_model))')  # Convertir en data.frame

# Convertir l'importance (qui est un R object) en DataFrame pandas
importance_df = pandas2ri.rpy2py(importance_r)

# Obtenir les prédictions en R et les convertir en liste (non numpy)
y_pred_r = ro.r('predict(rf_model, df_r)')
y_pred_list = list(y_pred_r)  # Convertir en liste

# Convertir les prédictions en DataFrame pandas
y_pred_df = pd.DataFrame(y_pred_list, columns=["Predictions"])

# Obtenir les vraies valeurs (y_true) directement depuis df_r
y_true_r = ro.r('df_r$area')

# Créer un DataFrame pour les vraies valeurs (y_true) et les prédictions (y_pred)
y_true_df = pd.DataFrame(y_true)
y_true_df["Predictions"] = y_pred_df

# Tracer le graphique directement en R
ro.r('''
    # Convertir les valeurs en R pour plot
    y_true <- c({0})
    y_pred <- c({1})

    # Créer le graphique
    plot(y_true, y_pred, main="Comparaison entre area_true et area_pred en fonction de DMC et DC area non nulle", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x pour la comparaison
'''.format(','.join(map(str, y_true_r)), ','.join(map(str, y_pred_r))))

# Sauvegarder le graphique dans un fichier PNG
file_path = "18_viken_m2icdsd_2025_b2_area_DMC_DC_prediction_vs_true_area_non_nulle.png"

# Sauvegarder la courbe dans un fichier PNG
ro.r(f'''
    png("{file_path}", width=800, height=600)
    plot(y_true, y_pred, main="Comparaison entre area_true et area_pred en fonction de DMC et DC area non nulle", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x
    dev.off()
''')

# Sauvegarder les résultats dans un fichier texte
with open("19_viken_m2icdsd_2025_b2_essai_random_forest_results_et_pred_area_tilde_DMC_DC_area_non_nulle.txt", "w", encoding="utf-8") as file:
    file.write("================= Résultats du modèle Random Forest area tilde DMC DC =================\n")
    file.write("\nImportance des variables :\n")
    file.write(importance_df.to_string())  # Écrire l'importance des variables dans le fichier
    file.write("\n\n===== Prédictions du modèle =====\n")
    file.write(y_true_df.to_string())  # Écrire les vraies valeurs et les prédictions dans le fichier


print("===========================================")

# Entraînement d'un modèle random forest pour prédire le FFMC en fonction de "temp" et "RH",
# détermination de l'importance de chaque variable dans la prédiction du FFMC
# puis comparer les prédictions du modèle avec les vraies valeurs du FFMC

# Activer la conversion automatique Pandas -> R DataFrame
pandas2ri.activate()

# Sélectionner uniquement les colonnes nécessaires
df_selected = df_area_non_0[["FFMC", "temp", "RH"]]

# Ajouter la colonne des vraies valeurs "ISI" dans le DataFrame
y_true = df_selected["FFMC"]

# Convertir en DataFrame R et assigner dans l'environnement R
df_r = pandas2ri.py2rpy(df_selected)
ro.globalenv["df_r"] = df_r  # Assigner df_r dans l'environnement R
ro.globalenv["y_true"] = pandas2ri.py2rpy(y_true)  # Assigner y_true dans l'environnement R

# Charger la bibliothèque randomForest en R
randomForest = importr("randomForest")

# Exécuter le Random Forest en R
rf_model = ro.r('''
    set.seed(123)  # Fixer le seed pour la reproductibilité
    rf_model <- randomForest(FFMC ~ temp + RH, data=df_r, ntree=500, importance=TRUE)
    print(rf_model)
    rf_model
''')

# Obtenir l'importance des variables
importance_r = ro.r('importance(rf_model)')
print(importance)

# Obtenir l'importance des variables sous forme de data.frame R
importance_r = ro.r('as.data.frame(importance(rf_model))')  # Convertir en data.frame

# Convertir l'importance (qui est un R object) en DataFrame pandas
importance_df = pandas2ri.rpy2py(importance_r)

# Obtenir les prédictions en R et les convertir en liste (non numpy)
y_pred_r = ro.r('predict(rf_model, df_r)')
y_pred_list = list(y_pred_r)  # Convertir en liste

# Convertir les prédictions en DataFrame pandas
y_pred_df = pd.DataFrame(y_pred_list, columns=["Predictions"])

# Obtenir les vraies valeurs (y_true) directement depuis df_r
y_true_r = ro.r('df_r$FFMC')

# Créer un DataFrame pour les vraies valeurs (y_true) et les prédictions (y_pred)
y_true_df = pd.DataFrame(y_true)
y_true_df["Predictions"] = y_pred_df

# Tracer le graphique directement en R
ro.r('''
    # Convertir les valeurs en R pour plot
    y_true <- c({0})
    y_pred <- c({1})

    # Créer le graphique
    plot(y_true, y_pred, main="Comparaison entre FFMC_true et FFMC_pred en fonction de temp et RH avec random forest", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x pour la comparaison
'''.format(','.join(map(str, y_true_r)), ','.join(map(str, y_pred_r))))

# Sauvegarder le graphique dans un fichier PNG
file_path = "20_viken_m2icdsd_2025_b2_FFMC_tilde_temp_RH_prediction_vs_true_random_forest.png"

# Sauvegarder la courbe dans un fichier PNG
ro.r(f'''
    png("{file_path}", width=800, height=600)
    plot(y_true, y_pred, main="Comparaison entre FFMC_true et FFMC_pred en fonction de temp et RH avec random forest", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x
    dev.off()
''')

# Sauvegarder les résultats dans un fichier texte
with open("21_viken_m2icdsd_2025_b2_essai_random_forest_results_et_pred_FFMC_tilde_temp_RH.txt", "w", encoding="utf-8") as file:
    file.write("================= Résultats du modèle Random Forest FFMC tilde temp RH =================\n")
    file.write("\nImportance des variables :\n")
    file.write(importance_df.to_string())  # Écrire l'importance des variables dans le fichier
    file.write("\n\n===== Prédictions du modèle =====\n")
    file.write(y_true_df.to_string())  # Écrire les vraies valeurs et les prédictions dans le fichier

print("===========================================")

# Entraînement d'un modèle random forest pour prédire le DC en fonction de "wind" et "RH",
# détermination de l'importance de chaque variable dans la prédiction du DC
# puis comparer les prédictions du modèle avec les vraies valeurs du DC

# Activer la conversion automatique Pandas -> R DataFrame
pandas2ri.activate()

# Sélectionner uniquement les colonnes nécessaires
df_selected = df_area_non_0[["DC", "wind", "RH"]]

# Ajouter la colonne des vraies valeurs 'ISI' dans le DataFrame
y_true = df_selected["DC"]

# Convertir en DataFrame R et assigner dans l'environnement R
df_r = pandas2ri.py2rpy(df_selected)
ro.globalenv["df_r"] = df_r  # Assigner df_r dans l'environnement R
ro.globalenv["y_true"] = pandas2ri.py2rpy(y_true)  # Assigner y_true dans l'environnement R

# Charger la bibliothèque randomForest en R
randomForest = importr("randomForest")

# Exécuter le Random Forest en R
rf_model = ro.r('''
    set.seed(123)  # Fixer le seed pour la reproductibilité
    rf_model <- randomForest(DC ~ wind + RH, data=df_r, ntree=500, importance=TRUE)
    print(rf_model)
    rf_model
''')

# Obtenir l'importance des variables
importance_r = ro.r('importance(rf_model)')
print(importance)

# Obtenir l'importance des variables sous forme de data.frame R
importance_r = ro.r('as.data.frame(importance(rf_model))')  # Convertir en data.frame

# Convertir l'importance (qui est un R object) en DataFrame pandas
importance_df = pandas2ri.rpy2py(importance_r)

# Obtenir les prédictions en R et les convertir en liste (non numpy)
y_pred_r = ro.r('predict(rf_model, df_r)')
y_pred_list = list(y_pred_r)  # Convertir en liste

# Convertir les prédictions en DataFrame pandas
y_pred_df = pd.DataFrame(y_pred_list, columns=["Predictions"])

# Obtenir les vraies valeurs (y_true) directement depuis df_r
y_true_r = ro.r('df_r$DC')

# Créer un DataFrame pour les vraies valeurs (y_true) et les prédictions (y_pred)
y_true_df = pd.DataFrame(y_true)
y_true_df["Predictions"] = y_pred_df

# Tracer le graphique directement en R
ro.r('''
    # Convertir les valeurs en R pour plot
    y_true <- c({0})
    y_pred <- c({1})

    # Créer le graphique
    plot(y_true, y_pred, main="Comparaison entre DC_true et DC_pred en fonction de wind et RH avec random forest", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x pour la comparaison
'''.format(','.join(map(str, y_true_r)), ','.join(map(str, y_pred_r))))

# Sauvegarder le graphique dans un fichier PNG
file_path = "22_viken_m2icdsd_2025_b2_random_forest_DC_tilde_wind_RH_prediction_vs_true.png"

# Sauvegarder la courbe dans un fichier PNG
ro.r(f'''
    png("{file_path}", width=800, height=600)
    plot(y_true, y_pred, main="Comparaison entre DC_true et DC_pred en fonction de wind et RH avec random forest", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x
    dev.off()
''')

# Sauvegarder les résultats dans un fichier texte
with open("23_viken_m2icdsd_2025_b2_essai_random_forest_results_et_pred_DC_tilde_wind_RH.txt", "w", encoding="utf-8") as file:
    file.write("================= Résultats du modèle Random Forest DC tilde wind RH =================\n")
    file.write("\nImportance des variables :\n")
    file.write(importance_df.to_string())  # Écrire l'importance des variables dans le fichier
    file.write("\n\n===== Prédictions du modèle =====\n")
    file.write(y_true_df.to_string())  # Écrire les vraies valeurs et les prédictions dans le fichier

# Entraînement d'un modèle random forest pour prédire la surface brûlée (version transformation logarithmique) log_area en fonction de
# "temp", "RH", "wind", "BUI", "FWI", "ISI", "DC" pour détermination de l'importance de chaque variable dans la prédiction de log_area
# puis comparer les prédictions du modèle avec les vraies valeurs de log_area

print("==========Essai de régression entre 'log_area' et 'temperature', 'RH', 'wind', 'BUI', 'FWI', 'ISI', 'DC' avec random forest=============")

# Activer la conversion automatique Pandas -> R DataFrame
pandas2ri.activate()

# Sélectionner uniquement les colonnes nécessaires
df_selected = df_area_non_0[["log_area", "temp", "RH", "wind", "BUI", "FWI", "ISI", "DC"]]

# Ajouter la colonne des vraies valeurs 'ISI' dans le DataFrame
y_true = df_selected["log_area"]

# Convertir en DataFrame R et assigner dans l'environnement R
df_r = pandas2ri.py2rpy(df_selected)
ro.globalenv["df_r"] = df_r
ro.globalenv["y_true"] = pandas2ri.py2rpy(y_true)  # Assigner y_true dans l'environnement R

# Charger la bibliothèque randomForest en R
randomForest = importr("randomForest")

# Exécuter le Random Forest en R
rf_model = ro.r('''
    set.seed(123)  # Fixer le seed pour la reproductibilité
    rf_model <- randomForest(log_area ~ temp + RH + wind + BUI + FWI + ISI + DC, 
                             data=df_r, ntree=500, importance=TRUE)
    print(rf_model)
    rf_model
''')

# Obtenir l'importance des variables
importance_r = ro.r('importance(rf_model)')
print(importance)

# Obtenir l'importance des variables sous forme de data.frame R
importance_r = ro.r('as.data.frame(importance(rf_model))')  # Convertir en data.frame

# Convertir l'importance (qui est un R object) en DataFrame pandas
importance_df = pandas2ri.rpy2py(importance_r)

# Obtenir les prédictions en R et les convertir en liste (non numpy)
y_pred_r = ro.r('predict(rf_model, df_r)')
y_pred_list = list(y_pred_r)  # Convertir en liste

# Convertir les prédictions en DataFrame pandas
y_pred_df = pd.DataFrame(y_pred_list, columns=["Predictions"])

# Obtenir les vraies valeurs (y_true) directement depuis df_r
y_true_r = ro.r('df_r$log_area')

# Créer un DataFrame pour les vraies valeurs (y_true) et les prédictions (y_pred)
y_true_df = pd.DataFrame(y_true)
y_true_df["Predictions"] = y_pred_df

# Tracer le graphique directement en R (valeurs prédites vs réelles)
ro.r('''
    # Convertir les valeurs en R pour plot
    y_true <- c({0})
    y_pred <- c({1})

    # Créer le graphique
    plot(y_true, y_pred, main="Comparaison entre log_area_true et log_area_pred avec meteo et indices en random forest", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x pour la comparaison
'''.format(','.join(map(str, y_true_r)), ','.join(map(str, y_pred_r))))

# Sauvegarder le graphique dans un fichier PNG
file_path = "24_viken_m2icdsd_2025_b2_random_forest_log_area_tilde_meteo_indices_prediction_vs_true.png"

# Sauvegarder la courbe dans un fichier PNG
ro.r(f'''
    png("{file_path}", width=800, height=600)
    plot(y_true, y_pred, main="Comparaison entre log_area_true et log_area_pred avec meteo et indices en random forest", 
         xlab="Valeurs réelles (y_true)", ylab="Valeurs prédites (y_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y=x
    dev.off()
''')

# Sauvegarder les résultats dans un fichier texte
with open("25_viken_m2icdsd_2025_b2_essai_random_forest_results_et_pred_log_area_tilde_meteo_indices.txt", "w", encoding="utf-8") as file:
    file.write("================= Résultats du modèle Random Forest log_area tilde meteo indices =================\n")
    file.write("\nImportance des variables :\n")
    file.write(importance_df.to_string())  # Écrire l'importance des variables dans le fichier
    file.write("\n\n===== Prédictions du modèle =====\n")
    file.write(y_true_df.to_string())  # Écrire les vraies valeurs et les prédictions dans le fichier

# Test d'une régression GAM (modèle additif généralisé) pour prédire la surface brûlée "area" à partir de "DMC" et "DC"
# puis courbe de comparaison entre valeur prédite et valeur réelle

# Activer la conversion automatique Pandas -> R DataFrame
pandas2ri.activate()

# Chargerl es bibliothèques nécessaires
mgcv = importr("mgcv")

# Sélectionner uniquement les colonnes nécessaires
df_selected = df_area_non_0[["area", "DMC", "DC"]]

# Convertir en DataFrame R et assigner dans l'environnement R
df_r = pandas2ri.py2rpy(df_selected)
ro.globalenv["df_r"] = df_r

# Exécuter le modèle GAM en R
gam_model = ro.r('''
    set.seed(123)  # Fixer le seed pour la reproductibilité
    gam_model <- gam(area ~ s(DMC) + s(DC), data=df_r, family=gaussian)
    gam_model
''')

# Obtenir le résumé du modèle (coefficients, p-values, R², etc.)
gam_summary = ro.r('summary(gam_model)')

# Obtenir l'analyse de la variance (ANOVA) pour voir l'importance des variables
anova_result = ro.r('anova(gam_model)')

# Obtenir les prédictions en R
y_pred_r = ro.r('predict(gam_model, df_r)')

# Convertir les prédictions en DataFrame pandas
y_pred_df = pd.DataFrame(list(y_pred_r), columns=["Predictions"])

# Obtenir les vraies valeurs (y_true) directement depuis df_selected
y_true_df = df_selected[["area"]].reset_index(drop=True)
y_true_df["Predictions"] = y_pred_df  # Ajouter les prédictions

# Sauvegarder les résultats dans un fichier texte
file_path_txt = "26_viken_m2icdsd_2025_b2_gam_results_predictions_area_tilde_DMC_DC_area_non_nulle.txt"
with open(file_path_txt, "w", encoding="utf-8") as file:
    file.write("================= Résultats du modèle GAM area ~ DMC + DC =================\n\n")
    file.write("=== Résumé du modèle GAM ===\n")
    file.write(str(gam_summary) + "\n\n")
    file.write("=== Analyse de la variance (ANOVA) ===\n")
    file.write(str(anova_result) + "\n\n")
    file.write("=== Prédictions du modèle ===\n")
    file.write(y_true_df.to_string())  # Écrire les vraies valeurs et les prédictions

print(f"Résultats enregistrés dans {file_path_txt}")

# Tracer le graphique directement en R et l'enregistrer
file_path_png = "27_viken_m2icdsd_2025_b2_gam_prediction_vs_true_area_tilde_DMC_DC_area_non_nulle.png"
ro.r(f'''
    png("{file_path_png}", width=800, height=600)
    plot(df_r$area, predict(gam_model, df_r), 
         main="Comparaison entre area_true et area_pred avec GAM, DMC et DC area non nulle",
         xlab="Valeurs réelles (area_true)", ylab="Valeurs prédites (area_pred)",
         pch=19, col='blue', cex=0.6)
    abline(a=0, b=1, col="red", lwd=2)  # Ajouter la ligne y = x pour la comparaison
    dev.off()
''')

print(f"Graphique enregistré sous {file_path_png}")

print("============== Essai Clustering hiérarchique====================")
print("=================Clustering Dward avec X,Y,ISI,FWI,area non nulle pour analyse répartition spatiale des feux======================")

# Sélectionner les colonnes nécessaires
df_selected = df_area_non_0[["X", "Y", "ISI", "FWI", "area"]]

# Convertir en DataFrame R
df_r = pandas2ri.py2rpy(df_selected)
ro.globalenv["df_r"] = df_r

# Clustering hiérarchique en R
ro.r('''
    # Normalisation des données
    df_scaled <- scale(df_r)

    # Clustering hiérarchique avec méthode de Ward
    hc <- hclust(dist(df_scaled), method = "ward.D2")

    # Couper l'arbre pour obtenir 3 clusters
    clusters <- cutree(hc, k = 3)

    # Ajouter les clusters au DataFrame
    df_r$cluster <- as.factor(clusters)

    # Tracer le dendrogramme et sauvegarder en fichier PNG
    png("28_viken_m2icdsd_2025_b2_dendrogramme_clusteringDward_area_non_0_sanstempRHwind.png", width = 800, height = 600)
    plot(hc, main = "Dendrogramme du Clustering Hiérarchique", xlab = "", sub = "", cex = 0.9)
    dev.off()
''')

# Sauvegarder le dendrogramme
print("Le dendrogramme a été sauvegardé dans le fichier '28_viken_m2icdsd_2025_b2_dendrogramme_clusteringDward_area_non_0_sanstempRHwind.png'.")

# Visualiser les clusters avec un scatter plot
ro.r('''

    # Vérifier la distribution des clusters
    print(table(df_r$cluster))
    
    # Créer un graphique de dispersion des points avec leurs clusters
    p <- ggplot(df_r, aes(x = X, y = Y, color = cluster)) +
        geom_point(size = 3, alpha = 0.7) +
        labs(title = "Visualisation des Clusters en fonction de X et Y area_non_0_sanstempRHwind",
             x = "Coordonnée X", y = "Coordonnée Y") +
        theme_minimal() +
        scale_color_manual(values = c("red", "green", "blue")) +
        
        theme(
            plot.title = element_text(color = "white", face = "bold", size = 14),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        )
        

    # Sauvegarder le scatter plot dans un fichier PNG
    ggsave("29_viken_m2icdsd_2025_b2_visualisation_clustersDward_white_area_non_0_sanstempRHwind.png", plot = p, width = 8, height = 6)
''')

# Message de confirmation pour la visualisation
print(f"Le graphique des clusters a été sauvegardé.")

# Code pour obtenir les statistiques descriptives par cluster
ro.r('''
    # Statistiques descriptives pour chaque cluster
    summary_cluster <- by(df_r[, c("X", "Y", "ISI", "FWI", "area")], df_r$cluster, summary)
    print(summary_cluster)
    # Récupérer les statistiques descriptives sous forme de texte
    summary_cluster_text <- capture.output(summary_cluster)
''')
# Récupérer les statistiques descriptives sous forme de texte
summary_cluster_r = ro.r['summary_cluster_text']

# Convertir les résultats en Python (ils sont maintenant sous forme de liste de chaînes de caractères)
summary_cluster_list = list(summary_cluster_r)

# Code pour calculer les moyennes des variables par cluster
ro.r('''
    # Moyennes des variables par cluster
    mean_by_cluster <- aggregate(df_r[, c("X", "Y", "ISI", "FWI", "area")], by = list(cluster = df_r$cluster), FUN = mean)
    print(mean_by_cluster)
''')
mean_by_cluster_r = ro.r['mean_by_cluster']

# Convertir mean_by_cluster en dataframe pandas
mean_by_cluster_df = pandas2ri.rpy2py(mean_by_cluster_r)

# Code pour générer et sauvegarder les boxplots pour chaque variable
# Code pour créer et sauvegarder les boxplots par cluster
ro.r('''
    # Charger le package ggplot2
    library(ggplot2)

    # Boxplot pour ISI
    p1 <- ggplot(df_r, aes(x = as.factor(cluster), y = ISI, fill = as.factor(cluster))) +
        geom_boxplot() +
        labs(title = "Distribution de ISI par cluster Dward area_non_0_sanstempRHwind", x = "Cluster", y = "ISI") +
        theme_minimal() +
        theme(
            plot.title = element_text(color = "white", face = "bold", size = 16),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        ) 

    # Sauvegarder le boxplot ISI
    ggsave("30_viken_m2icdsd_2025_b2_boxplot_isi_by_clusterDward_white_area_non_0_sanstempRHwind.png", plot = p1, width = 8, height = 6)

    # Boxplot pour FWI
    p2 <- ggplot(df_r, aes(x = as.factor(cluster), y = FWI, fill = as.factor(cluster))) +
        geom_boxplot() +
        labs(title = "Distribution de FWI par cluster area_non_0_sanstempRHwind", x = "Cluster", y = "FWI") +
        theme_minimal() +
        theme(
            plot.title = element_text(color = "white", face = "bold", size = 16),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        ) 

    # Sauvegarder le boxplot FWI
    ggsave("31_viken_m2icdsd_2025_b2_boxplot_fwi_by_clusterDward_white_area_non_0_sanstempRHwind.png", plot = p2, width = 8, height = 6)

    # Boxplot pour Surface Brûlée (area)
    p3 <- ggplot(df_r, aes(x = as.factor(cluster), y = area, fill = as.factor(cluster))) +
        geom_boxplot() +
        labs(title = "Distribution de la Surface Brûlée par cluster Dward area_non_0_sanstempRHwind", x = "Cluster", y = "Surface Brûlée (area)") +
        theme_minimal() +
        theme(
            plot.title = element_text(color = "white", face = "bold", size = 13),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        ) 

    # Sauvegarder le boxplot Surface Brûlée
    ggsave("32_viken_m2icdsd_2025_b2_boxplot_area_by_clusterDward_white_area_non_0_sanstempRHwind.png", plot = p3, width = 8, height = 6)
''')

# Message de confirmation
print("Les boxplots ont été sauvegardés sous les noms suivants :")
print("1. Boxplot ISI : 29_viken_m2icdsd_2025_b2_boxplot_isi_by_clusterDward_area_non_0_sanstempRHwind.png")
print("2. Boxplot FWI : 31_viken_m2icdsd_2025_b2_boxplot_fwi_by_clusterDward_area_non_0_sanstempRHwind.png")
print("3. Boxplot Surface Brûlée : 32_viken_m2icdsd_2025_b2_boxplot_area_by_clusterDward_area_non_0_sanstempRHwind.png")

# analyse des points du cluster 3
ro.r('''
    # Extraire les points du cluster 3
    cluster_3_points <- subset(df_r, cluster == 3)
    print(cluster_3_points)
    
    # Analyser les distances entre points
    dist_points <- dist(cluster_3_points[, c("X", "Y", "ISI", "FWI", "area")])
    
    # Convertir en matrice pour une meilleure gestion en Python
    dist_points_matrix <- as.matrix(dist_points)
    
    # Convertir la matrice en dataframe (facilite la conversion en pandas)
    dist_points_df <- as.data.frame(dist_points_matrix)
    print(dist_points_df)
''')
dist_points_r = ro.r['dist_points_df']

# Convertir mean_by_cluster en dataframe pandas
dist_points_df = pandas2ri.rpy2py(dist_points_r)

# Sauvegarde des résultats dans un fichier texte
with open("33_viken_m2icdsd_2025_b2_statistiques_par_clusterDward_area_non_0_sanstemprhwind.txt", "w", encoding="utf-8") as file:
    file.write("===== Statistiques Descriptives par Cluster Dward (surface brûlée non nulle, sans temp, RH et wind) =====\n")
    file.write("\n".join(summary_cluster_list))  # Écrire les stats descriptives capturées
    file.write("\n\n===== Moyennes des Variables par Cluster Dward =====\n")
    file.write(mean_by_cluster_df.to_string())  # Écrire le tableau des moyennes
    file.write("\n\n===== Distances entre les Points du Cluster 3 =====\n")
    file.write(dist_points_df.to_string())  # Écrire les distances dans le fichier


print("=================Clustering ward avec X,Y,ISI,FWI, temp, RH, wind, area non nulle pour analyse répartition spatiale des feux =========================")

# Sélectionner les colonnes nécessaires
df_selected = df_area_non_0[["X", "Y", "ISI", "FWI", "temp", "RH", "wind", "area"]]

# Convertir en DataFrame R
df_r = pandas2ri.py2rpy(df_selected)
ro.globalenv["df_r"] = df_r

# Clustering hiérarchique en R
ro.r('''
    # Normalisation des données
    df_scaled <- scale(df_r)

    # Clustering hiérarchique avec méthode de Ward
    hc <- hclust(dist(df_scaled), method = "ward.D2")

    # Couper l'arbre pour obtenir 3 clusters
    clusters <- cutree(hc, k = 3)

    # Ajouter les clusters au DataFrame
    df_r$cluster <- as.factor(clusters)

    # Tracer le dendrogramme et sauvegarder en fichier PNG
    png("34_viken_m2icdsd_2025_b2_dendrogramme_clusteringDWard_areanon0_avecmeteo.png", width = 800, height = 600)
    plot(hc, main = "Dendrogramme du Clustering Hiérarchique area_non_0_avecmeteo", xlab = "", sub = "", cex = 0.9)
    dev.off()
''')

# Sauvegarder le dendrogramme
print("Le dendrogramme a été sauvegardé dans le fichier '34_viken_m2icdsd_2025_b2_dendrogramme_clusteringDward_area_non_0_avecmeteo.png'.")

# Visualiser les clusters avec un scatter plot
ro.r('''

    # Vérifier la distribution des clusters
    print(table(df_r$cluster))

    # Créer un graphique de dispersion des points avec leurs clusters
    p <- ggplot(df_r, aes(x = X, y = Y, color = cluster)) +
        geom_point(size = 3, alpha = 0.7) +
        labs(title = "Visualisation des Clusters en fonction de X et Y area_non_0_avecmeteo",
             x = "Coordonnée X", y = "Coordonnée Y") +
        theme_minimal() +
        scale_color_manual(values = c("red", "green", "blue")) +

        theme(
            plot.title = element_text(color = "white", face = "bold", size = 14),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        )


    # Sauvegarder le scatter plot dans un fichier PNG
    ggsave("35_viken_m2icdsd_2025_b2_visualisation_clustersDward_white_area_non_0_avecmeteo.png", plot = p, width = 8, height = 6)
''')

# Message de confirmation pour la visualisation
print(f"Le graphique des clusters a été sauvegardé.")

# Code pour obtenir les statistiques descriptives par cluster
summary_cluster = ro.r('''
    # Statistiques descriptives pour chaque cluster
    summary_cluster <- by(df_r[, c("X", "Y", "ISI", "FWI", "temp", "RH", "wind", "area")], df_r$cluster, summary)
    print(summary_cluster)
    # Récupérer les statistiques descriptives sous forme de texte
    summary_cluster_text <- capture.output(summary_cluster)
''')
# Récupérer les statistiques descriptives sous forme de texte
summary_cluster_r = ro.r['summary_cluster_text']

# Convertir les résultats en Python (ils sont maintenant sous forme de liste de chaînes de caractères)
summary_cluster_list = list(summary_cluster_r)

# Code pour calculer les moyennes des variables par cluster
mean_by_cluster = ro.r('''
    # Moyennes des variables par cluster
    mean_by_cluster <- aggregate(df_r[, c("X", "Y", "ISI", "FWI", "temp", "RH", "wind", "area")], by = list(cluster = df_r$cluster), FUN = mean)
    print(mean_by_cluster)
''')
mean_by_cluster_r = ro.r['mean_by_cluster']

# Convertir mean_by_cluster en dataframe pandas
mean_by_cluster_df = pandas2ri.rpy2py(mean_by_cluster_r)

# Code pour générer et sauvegarder les boxplots pour chaque variable
# Code pour créer et sauvegarder les boxplots par cluster
ro.r('''
    # Charger le package ggplot2
    library(ggplot2)

    # Boxplot pour ISI
    p1 <- ggplot(df_r, aes(x = as.factor(cluster), y = ISI, fill = as.factor(cluster))) +
        geom_boxplot() +
        labs(title = "Distribution de ISI par cluster Dward area_non_0_avecmeteo", x = "Cluster", y = "ISI") +
        theme_minimal() +
        theme(
            plot.title = element_text(color = "white", face = "bold", size = 16),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        ) 

    # Sauvegarder le boxplot ISI
    ggsave("36_viken_m2icdsd_2025_b2_boxplot_isi_by_clusterDward_white_area_non_0_avecmeteo.png", plot = p1, width = 8, height = 6)

    # Boxplot pour FWI
    p2 <- ggplot(df_r, aes(x = as.factor(cluster), y = FWI, fill = as.factor(cluster))) +
        geom_boxplot() +
        labs(title = "Distribution de FWI par cluster Dward area_non_0_avecmeteo", x = "Cluster", y = "FWI") +
        theme_minimal() +
        theme(
            plot.title = element_text(color = "white", face = "bold", size = 16),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        ) 

    # Sauvegarder le boxplot FWI
    ggsave("37_viken_m2icdsd_2025_b2_boxplot_fwi_by_clusterDward_white_area_non_0_avecmeteo.png", plot = p2, width = 8, height = 6)

    # Boxplot pour Surface Brûlée (area)
    p3 <- ggplot(df_r, aes(x = as.factor(cluster), y = area, fill = as.factor(cluster))) +
        geom_boxplot() +
        labs(title = "Distribution de la Surface Brûlée par cluster Dward area_non_0_avecmeteo", x = "Cluster", y = "Surface Brûlée (area)") +
        theme_minimal() +
        theme(
            plot.title = element_text(color = "white", face = "bold", size = 13),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        ) 

    # Sauvegarder le boxplot Surface Brûlée
    ggsave("38_viken_m2icdsd_2025_b2_boxplot_area_by_clusterDward_white_area_non_0_avecmeteo.png", plot = p3, width = 8, height = 6)
''')

# Message de confirmation
print("Les boxplots ont été sauvegardés sous les noms suivants :")
print("1. Boxplot ISI : 36_viken_m2icdsd_2025_b2_boxplot_isi_by_clusterDward_area_non_0_avecmeteo.png")
print("2. Boxplot FWI : 37_viken_m2icdsd_2025_b2_boxplot_fwi_by_clusterDward_area_non_0_avecmeteo.png")
print("3. Boxplot Surface Brûlée : 38_viken_m2icdsd_2025_b2_boxplot_area_by_clusterDward_area_non_0_avecmeteo.png")

# analyse des points du cluster 3
dist_points = ro.r('''
    # Extraire les points du cluster 3
    cluster_3_points <- subset(df_r, cluster == 3)
    print(cluster_3_points)

    # Analyser les distances entre points
    dist_points <- dist(cluster_3_points[, c("X", "Y", "ISI", "FWI", "temp", "RH", "wind", "area")])
    
    # Convertir en matrice pour une meilleure gestion en Python
    dist_points_matrix <- as.matrix(dist_points)
    
    # Convertir la matrice en dataframe (facilite la conversion en pandas)
    dist_points_df <- as.data.frame(dist_points_matrix)
    print(dist_points_df)
''')
dist_points_r = ro.r['dist_points_df']

# Convertir mean_by_cluster en dataframe pandas
dist_points_df = pandas2ri.rpy2py(dist_points_r)

# Sauvegarde des résultats dans un fichier texte
with open("39_viken_m2icdsd_2025_b2_statistiques_par_clusterDward_area_non_0_avec_meteo.txt", "w", encoding="utf-8") as file:
    file.write("===== Statistiques Descriptives par Cluster Dward (surface brûlée non nulle, avec temp, RH et wind) =====\n")
    file.write("\n".join(summary_cluster_list))  # Écrire les stats descriptives capturées
    file.write("\n\n===== Moyennes des Variables par Cluster Dward =====\n")
    file.write(mean_by_cluster_df.to_string())  # Écrire le tableau des moyennes
    file.write("\n\n===== Distances entre les Points du Cluster 3 =====\n")
    file.write(dist_points_df.to_string())  # Écrire les distances dans le fichier



print("=================Clustering Dward avec X,Y,ISI,FWI,area non nulle et area nulle pour analyse répartition spatiale des feux=========================")

# Sélectionner les colonnes nécessaires
df_selected = df_cleaned[["X", "Y", "ISI", "FWI", "area"]]

# Convertir en DataFrame R
df_r = pandas2ri.py2rpy(df_selected)
ro.globalenv["df_r"] = df_r

# Clustering hiérarchique en R
ro.r('''
    # Normalisation des données
    df_scaled <- scale(df_r)

    # Clustering hiérarchique avec méthode de Ward
    hc <- hclust(dist(df_scaled), method = "ward.D2")

    # Couper l'arbre pour obtenir 3 clusters
    clusters <- cutree(hc, k = 3)

    # Ajouter les clusters au DataFrame
    df_r$cluster <- as.factor(clusters)

    # Tracer le dendrogramme et sauvegarder en fichier PNG
    png("40_viken_m2icdsd_2025_b2_dendrogramme_clusteringDward_area_sanstempRHwind.png", width = 1500, height = 600)
    plot(hc, main = "Dendrogramme du Clustering Hiérarchique", xlab = "", sub = "", cex = 0.9)
    dev.off()
''')

# Sauvegarder le dendrogramme
print("Le dendrogramme a été sauvegardé dans le fichier '40_viken_m2icdsd_2025_b2_dendrogramme_clusteringDward_area_sanstempRHwind.png'.")

# Visualiser les clusters avec un scatter plot
ro.r('''

    # Vérifier la distribution des clusters
    print(table(df_r$cluster))

    # Créer un graphique de dispersion des points avec leurs clusters
    p <- ggplot(df_r, aes(x = X, y = Y, color = cluster)) +
        geom_point(size = 3, alpha = 0.7) +
        labs(title = "Visualisation des Clusters Dward en fonction de X et Y area_sanstempRHwind",
             x = "Coordonnée X", y = "Coordonnée Y") +
        theme_minimal() +
        scale_color_manual(values = c("red", "green", "blue")) +

        theme(
            plot.title = element_text(color = "white", face = "bold", size = 13),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        )

    # Sauvegarder le scatter plot dans un fichier PNG
    ggsave("41_viken_m2icdsd_2025_b2_visualisation_clustersDward_white_area_sanstempRHwind.png", plot = p, width = 8, height = 6)
''')

# Message de confirmation pour la visualisation
print(f"Le graphique des clusters a été sauvegardé.")

# Code pour obtenir les statistiques descriptives par cluster
summary_cluster = ro.r('''
    # Statistiques descriptives pour chaque cluster
    summary_cluster <- by(df_r[, c("X", "Y", "ISI", "FWI", "area")], df_r$cluster, summary)
    print(summary_cluster)
    # Récupérer les statistiques descriptives sous forme de texte
    summary_cluster_text <- capture.output(summary_cluster)
''')

# Récupérer les statistiques descriptives sous forme de texte
summary_cluster_r = ro.r['summary_cluster_text']

# Convertir les résultats en Python (ils sont maintenant sous forme de liste de chaînes de caractères)
summary_cluster_list = list(summary_cluster_r)

# Code pour calculer les moyennes des variables par cluster
df_means = ro.r('''
    # Moyennes des variables par cluster
    mean_by_cluster <- aggregate(df_r[, c("X", "Y", "ISI", "FWI", "area")], by = list(cluster = df_r$cluster), FUN = mean)
    print(mean_by_cluster)
''')
mean_by_cluster_r = ro.r['mean_by_cluster']

# Convertir mean_by_cluster en dataframe pandas
mean_by_cluster_df = pandas2ri.rpy2py(mean_by_cluster_r)

# Code pour générer et sauvegarder les boxplots pour chaque variable
# Code pour créer et sauvegarder les boxplots par cluster
ro.r('''
    # Charger le package ggplot2
    library(ggplot2)

    # Boxplot pour ISI
    p1 <- ggplot(df_r, aes(x = as.factor(cluster), y = ISI, fill = as.factor(cluster))) +
        geom_boxplot() +
        labs(title = "Distribution de ISI par cluster area_sanstempRHwind", x = "Cluster", y = "ISI") +
        theme_minimal() +
        theme(
            plot.title = element_text(color = "white", face = "bold", size = 16),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        ) 

    # Sauvegarder le boxplot ISI
    ggsave("42_viken_m2icdsd_2025_b2_boxplot_isi_by_clusterDward_white_area_sanstempRHwind.png", plot = p1, width = 8, height = 6)

    # Boxplot pour FWI
    p2 <- ggplot(df_r, aes(x = as.factor(cluster), y = FWI, fill = as.factor(cluster))) +
        geom_boxplot() +
        labs(title = "Distribution de FWI par cluster area_sanstempRHwind", x = "Cluster", y = "FWI") +
        theme_minimal() +
        theme(
            plot.title = element_text(color = "white", face = "bold", size = 16),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        ) 

    # Sauvegarder le boxplot FWI
    ggsave("43_viken_m2icdsd_2025_b2_boxplot_fwi_by_clusterDward_white_area_sanstempRHwind.png", plot = p2, width = 8, height = 6)

    # Boxplot pour Surface Brûlée (area)
    p3 <- ggplot(df_r, aes(x = as.factor(cluster), y = area, fill = as.factor(cluster))) +
        geom_boxplot() +
        labs(title = "Distribution de la Surface Brûlée par cluster Dward area_sanstempRHwind", x = "Cluster", y = "Surface Brûlée (area)") +
        theme_minimal() +
        theme(
            plot.title = element_text(color = "white", face = "bold", size = 13),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        ) 

    # Sauvegarder le boxplot Surface Brûlée
    ggsave("44_viken_m2icdsd_2025_b2_boxplot_area_by_clusterDward_white_area_sanstempRHwind.png", plot = p3, width = 8, height = 6)
''')

# Message de confirmation
print("Les boxplots ont été sauvegardés sous les noms suivants :")
print("1. Boxplot ISI : 42_viken_m2icdsd_2025_b2_boxplot_isi_by_cluster_Dward_area_sanstempRHwind.png")
print("2. Boxplot FWI : 43_viken_m2icdsd_2025_b2_boxplot_fwi_by_cluster_Dward_area_sanstempRHwind.png")
print("3. Boxplot Surface Brûlée : 44_viken_m2icdsd_2025_b2_boxplot_area_by_cluster_Dward_area_sanstempRHwind.png")

# analyse des points du cluster 3
dist_points = ro.r('''
    # Extraire les points du cluster 3
    cluster_3_points <- subset(df_r, cluster == 3)
    print(cluster_3_points)

    # Analyser les distances entre points
    dist_points <- dist(cluster_3_points[, c("X", "Y", "ISI", "FWI", "area")])
    
    # Convertir en matrice pour une meilleure gestion en Python
    dist_points_matrix <- as.matrix(dist_points)
    
    # Convertir la matrice en dataframe (facilite la conversion en pandas)
    dist_points_df <- as.data.frame(dist_points_matrix)
    print(dist_points_df)
''')
dist_points_r = ro.r['dist_points_df']

# Convertir mean_by_cluster en dataframe pandas
dist_points_df = pandas2ri.rpy2py(dist_points_r)

# Sauvegarde des résultats dans un fichier texte
with open("45_viken_m2icdsd_2025_b2_statistiques_par_clusterDward_area_sans_meteo.txt", "w", encoding="utf-8") as file:
    file.write("===== Statistiques Descriptives par Cluster Dward (surface brûlée nulle et non nulle sans temp, RH et wind) =====\n")
    file.write("\n".join(summary_cluster_list))  # Écrire les stats descriptives capturées
    file.write("\n\n===== Moyennes des Variables par Cluster Dward =====\n")
    file.write(mean_by_cluster_df.to_string())  # Écrire le tableau des moyennes
    file.write("\n\n===== Distances entre les Points du Cluster 3 =====\n")
    file.write(dist_points_df.to_string())  # Écrire les distances dans le fichier


print("=================Clustering Dward avec X,Y,ISI,FWI, temp, RH, wind, area non nulle et area nulle pour analyse répartition spatiale des feux=========================")

# Sélectionner les colonnes nécessaires
df_selected = df_cleaned[["X", "Y", "ISI", "FWI", "temp", "RH", "wind", "area"]]

# Convertir en DataFrame R
df_r = pandas2ri.py2rpy(df_selected)
ro.globalenv["df_r"] = df_r

# Clustering hiérarchique en R
ro.r('''
    # Normalisation des données
    df_scaled <- scale(df_r)

    # Clustering hiérarchique avec méthode de Ward
    hc <- hclust(dist(df_scaled), method = "ward.D2")

    # Couper l'arbre pour obtenir 3 clusters
    clusters <- cutree(hc, k = 3)

    # Ajouter les clusters au DataFrame
    df_r$cluster <- as.factor(clusters)

    # Tracer le dendrogramme et sauvegarder en fichier PNG
    png("46_viken_m2icdsd_2025_b2_dendrogramme_clusteringDward_area_avecmeteo.png", width = 800, height = 600)
    plot(hc, main = "Dendrogramme du Clustering Hiérarchique Dward", xlab = "", sub = "", cex = 0.9)
    dev.off()
''')

# Sauvegarder le dendrogramme
print("Le dendrogramme a été sauvegardé dans le fichier '46_viken_m2icdsd_2025_b2_dendrogramme_clusteringDward_area_avecmeteo.png'.")

# Visualiser les clusters avec un scatter plot
ro.r('''

    # Vérifier la distribution des clusters
    print(table(df_r$cluster))

    # Créer un graphique de dispersion des points avec leurs clusters
    p <- ggplot(df_r, aes(x = X, y = Y, color = cluster)) +
        geom_point(size = 3, alpha = 0.7) +
        labs(title = "Visualisation des Clusters Dward en fonction de X et Y area avec meteo",
             x = "Coordonnée X", y = "Coordonnée Y") +
        theme_minimal() +
        scale_color_manual(values = c("red", "green", "blue")) +

        theme(
            plot.title = element_text(color = "white", face = "bold", size = 16),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        )

    # Sauvegarder le scatter plot dans un fichier PNG
    ggsave("47_viken_m2icdsd_2025_b2_visualisation_clusters_Dward_white_area_avecmeteo.png", plot = p, width = 8, height = 6)
''')

# Message de confirmation pour la visualisation
print(f"Le graphique des clusters a été sauvegardé.")

# Code pour obtenir les statistiques descriptives par cluster
summary_cluster = ro.r('''
    # Statistiques descriptives pour chaque cluster
    summary_cluster <- by(df_r[, c("X", "Y", "ISI", "FWI", "temp", "RH", "wind", "area")], df_r$cluster, summary)
    print(summary_cluster)
    # Récupérer les statistiques descriptives sous forme de texte
    summary_cluster_text <- capture.output(summary_cluster)
''')

# Récupérer les statistiques descriptives sous forme de texte
summary_cluster_r = ro.r['summary_cluster_text']

# Convertir les résultats en Python (ils sont maintenant sous forme de liste de chaînes de caractères)
summary_cluster_list = list(summary_cluster_r)

# Code pour calculer les moyennes des variables par cluster
df_means = ro.r('''
    # Moyennes des variables par cluster
    mean_by_cluster <- aggregate(df_r[, c("X", "Y", "ISI", "FWI", "temp", "RH", "wind", "area")], by = list(cluster = df_r$cluster), FUN = mean)
    print(mean_by_cluster)
''')
mean_by_cluster_r = ro.r['mean_by_cluster']

# Convertir mean_by_cluster en dataframe pandas
mean_by_cluster_df = pandas2ri.rpy2py(mean_by_cluster_r)

# Code pour générer et sauvegarder les boxplots pour chaque variable
# Code pour créer et sauvegarder les boxplots par cluster
ro.r('''
    # Charger le package ggplot2
    library(ggplot2)

    # Boxplot pour ISI
    p1 <- ggplot(df_r, aes(x = as.factor(cluster), y = ISI, fill = as.factor(cluster))) +
        geom_boxplot() +
        labs(title = "Distribution de ISI par cluster Dward area avec meteo", x = "Cluster", y = "ISI") +
        theme_minimal() +
        theme(
            plot.title = element_text(color = "white", face = "bold", size = 16),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        ) 

    # Sauvegarder le boxplot ISI
    ggsave("48_viken_m2icdsd_2025_b2_boxplot_isi_by_clusterDward_white_area_avecmeteo.png", plot = p1, width = 8, height = 6)

    # Boxplot pour FWI
    p2 <- ggplot(df_r, aes(x = as.factor(cluster), y = FWI, fill = as.factor(cluster))) +
        geom_boxplot() +
        labs(title = "Distribution de FWI par cluster Dward area avec meteo", x = "Cluster", y = "FWI") +
        theme_minimal() +
        theme(
            plot.title = element_text(color = "white", face = "bold", size = 16),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        ) 

    # Sauvegarder le boxplot FWI
    ggsave("49_viken_m2icdsd_2025_b2_boxplot_fwi_by_clusterDward_white_area_avecmeteo.png", plot = p2, width = 8, height = 6)

    # Boxplot pour Surface Brûlée (area)
    p3 <- ggplot(df_r, aes(x = as.factor(cluster), y = area, fill = as.factor(cluster))) +
        geom_boxplot() +
        labs(title = "Distribution de la Surface Brûlée par cluster Dward area avec meteo", x = "Cluster", y = "Surface Brûlée (area)") +
        theme_minimal() +
        theme(
            plot.title = element_text(color = "white", face = "bold", size = 16),  # Titre en blanc et en gras
            axis.title.x = element_text(color = "white", face = "bold", size = 12), # Titre axe X
            axis.title.y = element_text(color = "white", face = "bold", size = 12), # Titre axe Y
            axis.text.x = element_text(color = "white", size = 10),  # Texte axe X
            axis.text.y = element_text(color = "white", size = 10),  # Texte axe Y
            legend.title = element_text(color = "white", face = "bold", size = 12),  # Légende en blanc et en gras
            legend.text = element_text(color = "white", size = 10)  # Texte de légende
        ) 

    # Sauvegarder le boxplot Surface Brûlée
    ggsave("50_viken_m2icdsd_2025_b2_boxplot_area_by_clusterDward_white_area_avecmeteo.png", plot = p3, width = 8, height = 6)
''')

# Message de confirmation
print("Les boxplots ont été sauvegardés sous les noms suivants :")
print("1. Boxplot ISI : 48_viken_m2icdsd_2025_b2_boxplot_isi_by_clusterDward_area_avecmeteo.png")
print("2. Boxplot FWI : 49_viken_m2icdsd_2025_b2_boxplot_fwi_by_clusterDward_area_avecmeteo.png")
print("3. Boxplot Surface Brûlée : 50_viken_m2icdsd_2025_b2_boxplot_area_by_clusterDward_area_avecmeteo.png")

# analyse des points du cluster 3
dist_points = ro.r('''
    # Extraire les points du cluster 3
    cluster_3_points <- subset(df_r, cluster == 3)
    print(cluster_3_points)

    # Analyser les distances entre points
    dist_points <- dist(cluster_3_points[, c("X", "Y", "ISI", "FWI", "temp", "RH", "wind", "area")])
    
    # Convertir en matrice pour une meilleure gestion en Python
    dist_points_matrix <- as.matrix(dist_points)
    
    # Convertir la matrice en dataframe (facilite la conversion en pandas)
    dist_points_df <- as.data.frame(dist_points_matrix)
    print(dist_points_df)
''')
dist_points_r = ro.r['dist_points_df']

# Convertir mean_by_cluster en dataframe pandas
dist_points_df = pandas2ri.rpy2py(dist_points_r)

# Sauvegarde des résultats dans un fichier texte
with open("51_viken_m2icdsd_2025_b2_statistiques_par_clusterDward_area_avec_meteo.txt", "w", encoding="utf-8") as file:
    file.write(
        "===== Statistiques Descriptives par Cluster Dward (surface brûlée nulle et non nulle avec temp, RH et wind)  =====\n")
    file.write("\n".join(summary_cluster_list))  # Écrire les stats descriptives capturées
    file.write("\n\n===== Moyennes des Variables par Cluster Dward =====\n")
    file.write(mean_by_cluster_df.to_string())  # Écrire le tableau des moyennes
    file.write("\n\n===== Distances entre les Points du Cluster 3 =====\n")
    file.write(dist_points_df.to_string())  # Écrire les distances dans le fichier

print("============== Fin Essai Clustering hiérarchique====================")

# 4 Création et Alimentation de la Base de données traitée en utilisant PostGreSQL depuis Python

# Rappel des informations sur le dataframe traité à l'étape 2
print("Rappel des informations sur le dataframe df_cleaned")
print(df_cleaned.info())

# Convertir en chaînes de caractères les colonnes de type "category"
df_cleaned["day"] = df_cleaned["day"].astype(str)
df_cleaned["month"] = df_cleaned["month"].astype(str)
df_cleaned["season"] = df_cleaned["season"].astype(str)
df_cleaned["bscale"] = df_cleaned["bscale"].astype(str)
df_cleaned["bscale_description"] = df_cleaned["bscale_description"].astype(str)
df_cleaned["iscale"] = df_cleaned["iscale"].astype(str)
df_cleaned["iscale_description"] = df_cleaned["iscale_description"].astype(str)
df_cleaned["danger_level"] = df_cleaned["danger_level"].astype(str)
df_cleaned["level_description"] = df_cleaned["level_description"].astype(str)

# Connexion à PostgreSQL (connexion à la base par défaut 'postgres' pour créer une nouvelle base)
def create_database(database_name, user, password, host, port):
    connection = psycopg2.connect(
        dbname='postgres',
        user=user,
        password=password,
        host=host,
        port=port
    )
    connection.autocommit = True
    cursor = connection.cursor()

    try:
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
        print(f"La base de données '{database_name}' a été créée avec succès.")
    except psycopg2.errors.DuplicateDatabase:
        print(f"La base de données '{database_name}' existe déjà.")

    cursor.close()
    connection.close()

# Connexion à la base de données PostgreSQL avec encodage UTF-8
def connect_to_database(database_name, user, password, host, port):
    return psycopg2.connect(
        dbname=database_name,
        user=user,
        password=password,
        host=host,
        port=port,
        client_encoding='UTF8'  # confirmer encodage UTF-8
    )

# Créer la table dans la base de données
def create_table(cur):
    table_creation_query = """
    CREATE TABLE IF NOT EXISTS data_table_20002025 (
        id SERIAL PRIMARY KEY,
        X INTEGER,
        Y INTEGER,
        month VARCHAR,
        day VARCHAR,
        season VARCHAR,
        FFMC FLOAT,
        DMC FLOAT,
        DC FLOAT,
        ISI FLOAT,
        temp FLOAT,
        RH INTEGER,
        wind FLOAT,
        rain FLOAT,
        area FLOAT,
        log_area FLOAT,
        BUI FLOAT,
        FWI FLOAT,
        bscale VARCHAR,
        bscale_description VARCHAR,
        iscale VARCHAR,
        iscale_description VARCHAR,
        danger_level VARCHAR,
        level_description VARCHAR,
        CONSTRAINT unique_data UNIQUE (X, Y, month, day, season, FFMC, DMC, DC, ISI, temp, RH, wind, rain, area, log_area, BUI, FWI, bscale, bscale_description, iscale, iscale_description, danger_level, level_description)
    );
    """
    cur.execute(table_creation_query)

# Insérer les données du DataFrame dans la table
def insert_data_from_dataframe(cur, df):

    # Arrondir les colonnes "area" et "log_area" à un dixième
    df["log_area"] = df["log_area"].round(1).astype(float)
    df["area"] = df["area"].round(1).astype(float)

    # Convertir le DataFrame en une liste de tuples
    data_tuples = [tuple(row) for row in df[['X', 'Y', 'month', 'day', 'season', 'FFMC', 'DMC', 'DC', 'ISI',
                                             'temp', 'RH', 'wind', 'rain', 'area', 'log_area',
                                             'BUI', 'FWI', 'bscale', 'bscale_description',
                                             'iscale', 'iscale_description',
                                             'danger_level', 'level_description'
                                             ]].values]

    insert_query = """
    INSERT INTO data_table_20002025 (
        X, Y, month, day, season, FFMC, DMC, DC, ISI, temp, RH, wind, rain, area,
        log_area, BUI, FWI, bscale, bscale_description, iscale, iscale_description,
        danger_level, level_description
    ) VALUES %s
    ON CONFLICT (X, Y, month, day, season, FFMC, DMC, DC, ISI, temp, RH, wind, rain, 
        area, log_area, BUI, FWI, bscale, bscale_description, iscale, iscale_description, 
        danger_level, level_description)
    DO NOTHING;
    """

    # Insertion des données en une seule requête
    execute_values(cur, insert_query, data_tuples)

# Paramètres de connexion
database_name = "viken_db_20002025"  # Nom de votre base de données
user = "postgres"  # Votre utilisateur PostgreSQL
password = "formationviken"  # Votre mot de passe PostgreSQL
host = "localhost"  # Hôte PostgreSQL (ici localhost)
port = "5432"  # Port PostgreSQL

# Créer la base de données et insérer les données
try:
    # 1. Créer la base de données si elle n'existe pas
    create_database(database_name, user, password, host, port)

    # 2. Connecter à la base de données nouvellement créée
    conn = connect_to_database(database_name, user, password, host, port)
    cur = conn.cursor()

    # 3. Créer la table si elle n'existe pas
    create_table(cur)

    # 4. Insérer les données depuis le DataFrame
    insert_data_from_dataframe(cur, df_cleaned)

    # Commit des modifications
    conn.commit()

    print("Données insérées avec succès.")

    # 5. Exécuter une requête SQL q1 pour afficher la surface brûlée totale par saison
    cur.execute("""
                SELECT season, ROUND(SUM(area)::numeric, 1) AS total_area
                FROM data_table_20002025
                GROUP BY season
                ORDER BY SUM(area) DESC;
    """)

    # Récupérer les résultats
    results_q1 = cur.fetchall()

    # Afficher les résultats
    print("Superficie totale brûlée par saison :")
    for row in results_q1:
        print(row)

except Exception as e:
    print(f"Erreur d'exécution : {e}")

finally:
    # Fermer la connexion et le curseur
    if cur:
        cur.close()
    if conn:
        conn.close()


