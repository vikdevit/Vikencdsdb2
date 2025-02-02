import pandas as pd
import numpy as np
import rpy2.robjects
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Utiliser un backend sans interface graphique (pas de Tkinter)
import os
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
from scipy import stats
from scipy.stats import chi2_contingency
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr
#rpy2.robjects.r('install.packages("forecast")')
import psycopg2
from psycopg2 import sql
from psycopg2 import OperationalError
from psycopg2.extras import execute_values

# 1-Collecte, compréhension et audit de la qualité des données
# Chargement des données dans un dataframe
df = pd.read_csv("forestfires.csv")

# Affichage des premières lignes
print(df.head())

print("##################################################################################")

# Affichage des informations sur le dataframe
print(df.info())

print("##################################################################################")

# Conversion en float de la colonne area et contrôle des premières lignes de la colonne convertie area du dataframe
df["area"] = df["area"].str.replace(",", ".").astype(float)
print(f"Premières valeurs de la colonne area converties en float: ")
print(df["area"].head())

# Conversion en category des colonnes month et day
df['month'] = df['month'].astype('category')
df['day'] = df['day'].astype('category')
print(df.info())

print("##################################################################################")

# Affichage du nombre de valeurs manquantes pour chaque colonne
missing_counts = np.sum(df.isnull(), axis=0)

for col, count in zip(df.columns, missing_counts):
    print(f"Colonne {col}: {count} valeur(s) manquante(s)")

print("##################################################################################")

# Recherche des valeurs distinctes et de leur nombre pour les colonnes month et day
columns_to_check = ["month", "day"]

for column in columns_to_check:
    unique_values = np.unique(df[column])
    unique_count = len(unique_values)
    print(f"Colonne {column}:")
    print(f"  Valeurs distinctes: {unique_values}")
    print(f"  Nombre de valeurs distinctes: {unique_count}\n")

print("##################################################################################")

# Recherche de la valeur minimale et maximale des composants du FWI system, du vent, des précipitations et de la surface brûlée
columns_to_check = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"]

result_minmax = df[columns_to_check].agg(["min", "max"])
print(result_minmax)

print("##################################################################################")

# Recherche des valeurs des composants au-dessus de la plage théorique de valeurs

# Plage théorique pour chaque indice FWI selon https://confluence.ecmwf.int/display/CEMS/User+Guide
theoretical_ranges = {
    "FFMC": (0, 101),   #The FFMC ranges from 0 to 101, where higher values indicate drier and more easily ignitable fuels.
    "DMC": (0, 1000),   #The DMC ranges from 0 to 1000, with higher values indicating drier conditions.
    "DC": (0, 1000), #The DC ranges from 0 to 1000, with higher values indicating drier conditions.
    "ISI": (0, 50), #The ISI ranges from 0 to 50, with higher values indicating a faster fire spread potential.
}

# Liste des colonnes à vérifier
columns_to_check = ["FFMC", "DMC", "DC", "ISI"]

# Vérification des valeurs au-dessus de la plage théorique pour les indices FWI
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

# Dictionnaire des types attendus pour chaque colonne
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
        print(f"Colonnes avec valeurs non conformes dans '{column}':")
        print(non_conforming_rows, "\n")

# Si aucune ligne non conforme n'a été trouvée
if not has_non_conforming_rows:
    print("Aucune ligne trouvée de type non conforme avec la grandeur de la colonne.")

print("##################################################################################")

# Recherche des doublons et affichage des paires de lignes en doublon
duplicates = df[df.duplicated(keep=False)]  # Conserver toutes les occurrences des doublons

if duplicates.empty:
    print("Aucun doublon trouvé.")
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
pd.set_option('display.max_columns', None)
print(df.describe())

print("##################################################################################")

# 2-Alimentation, nettoyage et traitement des données

# Suppression des lignes en doublons
print("\nLignes à supprimer (indices) :")
print(to_drop)

df_cleaned = df.drop(to_drop)

# Affichage des informations du nouveau dataframe
df_cleaned.info()

print("##################################################################################")

# Création d'une nouvelle colonne 'season' pour stocker la saison correspondante
# Définition des saisons
season_mapping = {
    'jan': 'Hiver', 'feb': 'Hiver', 'dec': 'Hiver',
    'mar': 'Printemps', 'apr': 'Printemps', 'may': 'Printemps',
    'jun': 'Été', 'jul': 'Été', 'aug': 'Été',
    'sep': 'Automne', 'oct': 'Automne', 'nov': 'Automne'
}

df_cleaned['season'] = df_cleaned['month'].map(season_mapping).astype('category')
df_cleaned.info()

print("##################################################################################")

# Ajout d'une colonne calculant le BUI index pour en déduire le FWI index à partir de la formule de https://wikifire.wsl.ch/tiki-index8720.html?page=Buildup+index
def calculate_bui(dmc, dc):
    condition = dmc <= 0.4 * dc
    bui = np.where(
        condition,
        (0.8 * dmc * dc) / (dmc + 0.4 * dc),
        dmc - (1 - (0.8 * dc) / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
    )
    rounded_bui = np.round(bui, 1)
    return rounded_bui

df_cleaned['BUI'] = calculate_bui(df_cleaned['DMC'], df_cleaned['DC'])

# Caster la colonne 'BUI' en type 'float'
df_cleaned['BUI'] = df_cleaned['BUI'].astype('float')

# Vérifier le type de la colonne
print(df_cleaned['BUI'].dtype)

# Affichage du DataFrame avec la nouvelle colonne
print("Premieres lignes du dataframe avec colonne BUI")
print(df_cleaned.head())

# Ajout d'une colonne estimant le FWI index à partir d'une formule simplifiée selon publication de Van Wagner (1987)
def calculate_fwi(isi, bui):
    #return np.exp(0.05039 * isi) * bui ** 0.82 formule à vérifier
    fwi = np.sqrt(0.1 * isi * bui)
    rounded_fwi = np.round(fwi, 1)
    return rounded_fwi

df_cleaned['FWI'] = calculate_fwi(df_cleaned['ISI'], df_cleaned['BUI'])

# Caster la colonne 'FWI' en type 'float'
df_cleaned['FWI'] = df_cleaned['FWI'].astype('float')

# Vérifier le type de la colonne
print(df_cleaned['FWI'].dtype)

# Affichage du DataFrame avec les nouvelles colonnes
print("Premieres lignes du dataframe avec colonne FWI")
print(df_cleaned['FWI'])

# Désactiver la limitation d'affichage
pd.set_option('display.max_rows', None)

# Afficher la colonne FWI
print(df_cleaned['FWI'])

# Afficher la valeur max de la colonne FWI
print(f"La valeur maximale de FWI est: {df_cleaned['FWI'].max()}.")

# Ajouter un champ niveau de danger et un champ description du niveau selon https://climate-adapt.eea.europa.eu/en/metadata/indicators/fire-weather-index-monthly-mean-1979-2019
def get_danger_level(fwi):
    if fwi < 5.2:
        return "Très faible danger"
    elif 5.2 <= fwi < 11.2:
        return "Faible danger"
    elif 11.2 <= fwi < 21.3:
        return "Danger modéré"
    elif 21.3 <= fwi < 38.0:
        return "Fort danger"
    elif 38.0 <= fwi < 50.0:
        return "Très fort danger"
    else:
        return "Danger extrême"

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

# Ajouter les nouvelles colonnes 'danger_level' et 'level_description' au DataFrame
df_cleaned['danger_level'] = df_cleaned['FWI'].apply(get_danger_level)
df_cleaned['level_description'] = df_cleaned['FWI'].apply(get_level_description)

# Caster les colonnes 'danger_level' et 'level_description' en type 'category'
df_cleaned['danger_level'] = df_cleaned['danger_level'].astype('category')
df_cleaned['level_description'] = df_cleaned['level_description'].astype('category')

# Vérifier le type des nouvelles colonnes
print(df_cleaned['danger_level'].dtype)
print(df_cleaned['level_description'].dtype)

# Affichage du DataFrame avec les nouvelles colonnes
print(df_cleaned[['FWI', 'danger_level', 'level_description']])

print("##################################################################################")

# Comptage des lignes où 'area' est égal à 0 et différent de 0
count_area_0 = (df_cleaned['area'] == 0).sum()  # Nombre de lignes avec area == 0
count_area_non_0 = (df_cleaned['area'] != 0).sum()  # Nombre de lignes avec area != 0

# Créer un DataFrame pour ces comptages
data = {'Condition': ['area = 0', 'area != 0'], 'Count': [count_area_0, count_area_non_0]}
df_count = pd.DataFrame(data)

# Tracer le diagramme en barres
plt.figure(figsize=(6, 4))
sns.barplot(x='Condition', y='Count', data=df_count, palette='Blues')
plt.title("Nombre de lignes avec 'area' égal à 0 et différent de 0")
plt.xlabel("Condition")
plt.ylabel("Nombre de lignes")
plt.show()

print("##################################################################################")

# Ajout d'une colonne pour transformation logarithmique de 'area' pour améliorer la normalité
df_cleaned['log_area'] = np.log1p(df_cleaned['area'])

print("##################################################################################")

# Création de deux dataframe sous-ensembles de df_cleaned (area = 0 et area != 0)

# Filtrer les lignes où 'area' est égal à 0
df_area_0 = df_cleaned[df_cleaned['area'] == 0]

# Filtrer les lignes où 'area' est différent de 0
df_area_non_0 = df_cleaned[df_cleaned['area'] != 0]

# Afficher les DataFrames résultants
print("DataFrame avec 'area' = 0:")
print(df_area_0.head())

print("\nDataFrame avec 'area' != 0:")
print(df_area_non_0.head())

print("##################################################################################")

# 3-Analyse et Visualisation des données
# Analyses univariées
# 📁 Création du répertoire pour les analyses univariées
save_dir = "analyse_distrib_univariee"
os.makedirs(save_dir, exist_ok=True)

# Mesure de l'aplatissement de chaque distribution (kurtosis)
# Création du PDF
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

# Création du PDF
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

# Séparation des variables numériques et catégoriques
num_vars = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "BUI", "temp", "RH", "wind", "rain", "area", "log_area", "FWI"]
cat_vars = ["month", "day", "season", "danger_level", "level_description"]

# 📌 **1. Visualisation des variables numériques**
for col in num_vars:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_cleaned[col], bins=10, kde=True, color="royalblue")
    plt.title(f"Distribution de {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Fréquence", fontsize=12)

    # Sauvegarde des figures en PNG et PDF
    plt.savefig(os.path.join(save_dir, f"{col}.png"), format="png", dpi=300)
    plt.savefig(os.path.join(save_dir, f"{col}.pdf"), format="pdf", dpi=300)

    plt.close()  # Fermer la figure pour éviter l'affichage multiple

# 📌 **2. Visualisation des variables catégoriques**
for col in cat_vars:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df_cleaned[col], palette="viridis")
    plt.title(f"Répartition de {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Nombre d'observations", fontsize=12)
    plt.xticks(rotation=45)

    # Sauvegarde des figures en PNG et PDF
    plt.savefig(os.path.join(save_dir, f"{col}.png"), format="png", dpi=300)
    plt.savefig(os.path.join(save_dir, f"{col}.pdf"), format="pdf", dpi=300)

    plt.close()  # Fermer la figure pour éviter l'affichage multiple

print(f"Les graphiques sont enregistrés dans le dossier : {save_dir}")

# Affichage de plusieurs courbes sur une même page

# Création du répertoire pour les graphiques
#save_dir = "analyse_distrib_univariee_subplots"
#os.makedirs(save_dir, exist_ok=True)

# Configuration du style des graphes
sns.set_style("whitegrid")

### 📌 1. Graphiques X et Y sur la même figure ###
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
plt.savefig(os.path.join(save_dir, "X_Y.pdf"), format="pdf", dpi=300)
plt.close()

### 📌 2. Graphiques FFMC, DMC, DC, ISI, BUI, FWI sur la même figure ###
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 15))
fire_vars = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]

for i, col in enumerate(fire_vars):
    sns.histplot(df_cleaned[col], bins=10, kde=True, ax=axes[i // 2, i % 2], color="royalblue")
    axes[i // 2, i % 2].set_title(f"Distribution de {col}", fontsize=14)
    axes[i // 2, i % 2].set_xlabel(col, fontsize=12)
    axes[i // 2, i % 2].set_ylabel("Fréquence", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "FFMC_DMC_DC_ISI_BUI_FWI.png"), format="png", dpi=300)
plt.savefig(os.path.join(save_dir, "FFMC_DMC_DC_ISI_BUI_FWI.pdf"), format="pdf", dpi=300)
plt.close()

### 📌 3. Graphiques temp, RH, wind, rain sur la même figure ###
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
weather_vars = ["temp", "RH", "wind", "rain"]

for i, col in enumerate(weather_vars):
    sns.histplot(df_cleaned[col], bins=10, kde=True, ax=axes[i // 2, i % 2], color="darkorange")
    axes[i // 2, i % 2].set_title(f"Distribution de {col}", fontsize=14)
    axes[i // 2, i % 2].set_xlabel(col, fontsize=12)
    axes[i // 2, i % 2].set_ylabel("Fréquence", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "temp_RH_wind_rain.png"), format="png", dpi=300)
plt.savefig(os.path.join(save_dir, "temp_RH_wind_rain.pdf"), format="pdf", dpi=300)
plt.close()

print(f"Les graphiques sont enregistrés dans le dossier : {save_dir}")

# 📁 Création du répertoire pour sauvegarder les graphiques
#save_dir = "analyse_distrib_univariee_area"
#os.makedirs(save_dir, exist_ok=True)

# 📊 Comptage des valeurs où area = 0 et area > 0
df_count = pd.DataFrame({
    "Condition": ["Surface Brûlée = 0", "Surface Brûlée > 0"],
    "Count": [sum(df_cleaned["area"] == 0), sum(df_cleaned["area"] > 0)]
})

# 🔥 Tracer le diagramme en barres
plt.figure(figsize=(7, 5))
sns.barplot(x='Condition', y='Count', data=df_count, palette='Blues')

# 📌 Personnalisation du graphique
plt.title("Nombre de lignes avec 'area' égal à 0 et différent de 0", fontsize=14, fontweight='bold')
plt.xlabel("Condition", fontsize=12)
plt.ylabel("Nombre de lignes", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 📂 Sauvegarde du graphique
file_path_png = os.path.join(save_dir, "area_distribution.png")
file_path_pdf = os.path.join(save_dir, "area_distribution.pdf")
plt.savefig(file_path_png, format="png", dpi=300)
plt.savefig(file_path_pdf, format="pdf", dpi=300)
plt.close()

print("##################################################################################")

# Statistiques descriptives

# 📁 Création du répertoire pour sauvegarder les fichiers

save_dir = "desc_stats"
os.makedirs(save_dir, exist_ok=True)

# 📌 Liste des regroupements de variables pour les pages du PDF
plots_groups = {
    "X_Y": ["X", "Y"],
    "FFMC_DMC_DC_ISI_BUI_FWI": ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"],
    "Temp_RH_Wind_Rain": ["temp", "RH", "wind", "rain"],
    "Area": ["area", "log_area"]
}

# 📄 Création du fichier PDF
pdf_path = os.path.join(save_dir, "boxplots_with_stats.pdf")
with PdfPages(pdf_path) as pdf:
    # 🔥 Génération des boxplots
    for page_name, cols in plots_groups.items():
        fig, axes = plt.subplots(nrows=1, ncols=len(cols), figsize=(6 * len(cols), 6))

        # ✅ S'assurer que axes est toujours une liste
        if len(cols) == 1:
            axes = [axes]  # Convertir en liste si une seule variable

        # 🔄 Création des boxplots
        for i, col in enumerate(cols):
            ax = axes[i]
            sns.boxplot(y=df_cleaned[col], ax=ax, color="royalblue", width=0.5)

            # 📊 Calcul des statistiques
            Q1 = df_cleaned[col].quantile(0.25)
            Q2 = df_cleaned[col].median()  # Médiane
            Q3 = df_cleaned[col].quantile(0.75)
            Q4 = df_cleaned[col].max()
            mean_value = df_cleaned[col].mean()

            # 🎯 Ajout des valeurs sur le graphique
            statist = {
                "Q1": Q1,
                "Médiane (Q2)": Q2,
                "Q3": Q3,
                "Max (Q4)": Q4,
                "Moyenne": mean_value
            }

            # 📌 Positionner les textes sur le graphique
            for j, (stat_name, value) in enumerate(statist.items()):
                ax.text(0, value, f"{stat_name}: {value:.2f}", ha='center', va='bottom',
                        fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.6))

            # 🎨 Ajout du titre et labels
            ax.set_title(f"Boxplot de {col}", fontsize=14, fontweight='bold')
            ax.set_ylabel(col, fontsize=12)
            ax.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()

        # 📌 Sauvegarde en PDF
        pdf.savefig(fig)

        # 📌 Sauvegarde en PNG
        png_path = os.path.join(save_dir, f"boxplot_{page_name}.png")
        plt.savefig(png_path, dpi=300)

        plt.close()

print(f"✅ Boxplots sauvegardés dans {save_dir}.")

print("##################################################################################")

# 📏 Calcul des statistiques descriptives avec Numpy
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
    statistics[col] = statist

# 🔹 Création du DataFrame des statistiques
desc_stats_df = pd.DataFrame(statistics).T

# 📄 Création du fichier PDF et PNG pour sauvegarder
pdf_path = os.path.join(save_dir, "descriptive_statistics.pdf")
png_path = os.path.join(save_dir, "descriptive_statistics.png")

with PdfPages(pdf_path) as pdf:
    # 📊 Tracer les statistiques sous forme de tableau
    fig, ax = plt.subplots(figsize=(10, 6))  # Taille du graphique pour le tableau
    ax.axis('off')
    ax.table(cellText=desc_stats_df.values, colLabels=desc_stats_df.columns, rowLabels=desc_stats_df.index,
             loc='center',
             cellLoc='center', colLoc='center', bbox=[0, 0, 1, 1])

    # 📌 Sauvegarder le tableau en PDF
    pdf.savefig(fig)

    # 📌 Sauvegarder le tableau en PNG
    plt.savefig(png_path, dpi=300)

    plt.close()

print(f"✅ Statistiques descriptives sauvegardées dans {save_dir}.")

print("##################################################################################")

# Analyses bivariées

# Définir un répertoire et le créer si non encore existant
save_dir = "analyses_bivariées"
os.makedirs(save_dir, exist_ok=True)

# Visualisation de la distribution mensuelle de la surface brûlée sur le dataframe avec surface brûlée non nulle

# Liste des mois dans l'ordre du calendrier
mois_ordre = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

# 1. Boxplot de la distribution de la surface brûlée par mois
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_area_non_0, x='month', y='area', palette="Set2", order=mois_ordre)
plt.title("Distribution Mensuelle de la Surface Brûlée")
plt.xlabel("Mois")
plt.ylabel("Surface Brûlée (area)")
plt.xticks(rotation=45)
plt.tight_layout()

# Sauvegarde du graphique en .png et .pdf
boxplot_filename_png = os.path.join(save_dir, "boxplot_surface_brulee.png")
boxplot_filename_pdf = os.path.join(save_dir, "boxplot_surface_brulee.pdf")
plt.savefig(boxplot_filename_png)
plt.savefig(boxplot_filename_pdf)
plt.close()

# 2. Diagramme en barres pour la somme de la surface brûlée par mois
monthly_sum = df_area_non_0.groupby('month')['area'].sum().reset_index()  # Somme de la surface brûlée par mois
plt.figure(figsize=(10, 6))
sns.barplot(data=monthly_sum, x='month', y='area', palette="Set3", order=mois_ordre)
plt.title("Surface Brûlée Totale par Mois")
plt.xlabel("Mois")
plt.ylabel("Surface Brûlée Totale")
plt.xticks(rotation=45)
plt.tight_layout()

# Sauvegarde du graphique en .png et .pdf
barplot_sum_filename_png = os.path.join(save_dir, "barplot_sum_surface_brulee.png")
barplot_sum_filename_pdf = os.path.join(save_dir, "barplot_sum_surface_brulee.pdf")
plt.savefig(barplot_sum_filename_png)
plt.savefig(barplot_sum_filename_pdf)
plt.close()

# 3. Diagramme en barres pour la moyenne de la surface brûlée par mois
monthly_avg = df_area_non_0.groupby('month')['area'].mean().reset_index()  # Moyenne de la surface brûlée par mois
plt.figure(figsize=(10, 6))
sns.barplot(data=monthly_avg, x='month', y='area', palette="Set3", order=mois_ordre)
plt.title("Surface Brûlée Moyenne par Mois")
plt.xlabel("Mois")
plt.ylabel("Surface Brûlée Moyenne")
plt.xticks(rotation=45)
plt.tight_layout()

# Sauvegarde du graphique en .png et .pdf
barplot_avg_filename_png = os.path.join(save_dir, "barplot_avg_surface_brulee.png")
barplot_avg_filename_pdf = os.path.join(save_dir, "barplot_avg_surface_brulee.pdf")
plt.savefig(barplot_avg_filename_png)
plt.savefig(barplot_avg_filename_pdf)
plt.close()

print("Les graphiques ont été enregistrés dans le répertoire 'analyses_bivariees'.")

# Visualisation de la fréquence des incendies par mois
plt.figure(figsize=(10, 6))
sns.countplot(data=df_area_non_0, x='month', palette="Set2", order=mois_ordre)
plt.title("Fréquence des Incendies par Mois")
plt.xlabel("Mois")
plt.ylabel("Nombre d'Incendies")
plt.xticks(rotation=45)
plt.tight_layout()

# 📂 Sauvegarde du graphique
plt.savefig(f"{save_dir}/frequence_incendies_par_mois.png")
plt.savefig(f"{save_dir}/frequence_incendies_par_mois.pdf")
plt.close()

# Visualisation de la fréquence des incendies par saison
# Comptage du nombre d'incendies par saison
season_counts = df_area_non_0['season'].value_counts().reindex(['Hiver', 'Printemps', 'Été', 'Automne'])

# 📊 Création du diagramme en barres
plt.figure(figsize=(8, 6))
sns.barplot(x=season_counts.index, y=season_counts.values, palette="coolwarm")
plt.title("Fréquence des Incendies par Saison")
plt.xlabel("Saison")
plt.ylabel("Nombre d'Incendies")
plt.xticks(rotation=0)
plt.tight_layout()

# 📂 Sauvegarde du graphique en PNG et PDF
season_freq_png = os.path.join(save_dir, "frequence_incendies_saisons.png")
season_freq_pdf = os.path.join(save_dir, "frequence_incendies_saisons.pdf")
plt.savefig(season_freq_png)
plt.savefig(season_freq_pdf)
plt.close()

print("Le graphique de la fréquence des incendies par saison a été enregistré dans 'analyses_bivariees'.")

# Visualisation de la surface brûlée en fonction des coordonnées X et Y

# Créez une figure 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Définir les données
x = df_area_non_0['X']  # Coordonnée X
y = df_area_non_0['Y']  # Coordonnée Y
z = df_area_non_0['area']  # Surface brûlée (area)

# Créer un scatter plot en 3D
ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', edgecolors='k', alpha=0.7)

# Ajouter les titres et les labels
ax.set_title('Visualisation 3D de la Surface Brûlée en fonction de X et Y', fontsize=16)
ax.set_xlabel('Coordonnée X', fontsize=12)
ax.set_ylabel('Coordonnée Y', fontsize=12)
ax.set_zlabel('Surface Brûlée (area)', fontsize=12)

# Afficher la colorbar pour indiquer l'intensité des surfaces brûlées
cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Surface Brûlée', rotation=270, labelpad=20)

# Enregistrer le graphique en .png et .pdf dans le répertoire
png_path = os.path.join(save_dir, 'visualisation_surface_brulee.png')
pdf_path = os.path.join(save_dir, 'visualisation_surface_brulee.pdf')

fig.savefig(png_path, format='png', bbox_inches='tight')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')

# Fermer la figure après enregistrement
plt.close()

# Message de confirmation
print(f"Le graphique a été enregistré dans le répertoire : {save_dir}")
print(f"Fichier .png enregistré sous : {png_path}")
print(f"Fichier .pdf enregistré sous : {pdf_path}")

# Version avec surface de la visualisation de la surface brûlée en fonction des coordonnées X et Y
# Créez une figure 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Définir les données
x = df_area_non_0['X']  # Coordonnée X
y = df_area_non_0['Y']  # Coordonnée Y
z = df_area_non_0['area']  # Surface brûlée (area)

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
ax.set_title('Représentation de la Surface Brûlée en fonction de X et Y', fontsize=16)
ax.set_xlabel('Coordonnée X', fontsize=12)
ax.set_ylabel('Coordonnée Y', fontsize=12)
ax.set_zlabel('Surface Brûlée (area)', fontsize=12)

# Afficher la colorbar pour indiquer l'intensité des surfaces brûlées
cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Surface Brûlée', rotation=270, labelpad=20)

# Enregistrer le graphique en .png et .pdf dans le répertoire
png_path = os.path.join(save_dir, 'surface_brulee_3d.png')
pdf_path = os.path.join(save_dir, 'surface_brulee_3d.pdf')

# Enregistrer les fichiers .png et .pdf
fig.savefig(png_path, format='png', bbox_inches='tight')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')

# Fermer la figure après l'enregistrement pour éviter toute interférence
plt.close(fig)

# Afficher un message de confirmation
print(f"Le graphique a été enregistré dans le répertoire : {save_dir}")
print(f"Fichier .png enregistré sous : {png_path}")
print(f"Fichier .pdf enregistré sous : {pdf_path}")

# Visualiser la surface brûlée en fonction de la température
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_area_non_0, x='temp', y='area', alpha=0.6, edgecolor=None, color="royalblue")
plt.title("Surface Brûlée en Fonction de la Température")
plt.xlabel("Température (°C)")
plt.ylabel("Surface Brûlée (ha)")
plt.grid(True)
plt.tight_layout()

# 📂 Sauvegarde du graphique
plt.savefig(f"{save_dir}/surface_brulee_vs_temperature.png")
plt.savefig(f"{save_dir}/surface_brulee_vs_temperature.pdf")
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

# 📂 Sauvegarde du graphique
plt.savefig(f"{save_dir}/surface_brulee_et_surfacelog_vs_temperature.png")
plt.savefig(f"{save_dir}/surface_brulee_etsurfacelog_vs_temperature.pdf")
plt.close()

# Visualiser la surface brûlée en fonction de l'humidité relative
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_area_non_0, x='RH', y='area', alpha=0.6, edgecolor=None, color="royalblue")
plt.title("Surface Brûlée en Fonction de l'humidité relative")
plt.xlabel("RH (%)")
plt.ylabel("Surface Brûlée (ha)")
plt.grid(True)
plt.tight_layout()

# 📂 Sauvegarde du graphique
plt.savefig(f"{save_dir}/surface_brulee_vs_humidite_relative.png")
plt.savefig(f"{save_dir}/surface_brulee_vs_humidite_relative.pdf")
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
plt.title("Surface Brûlée et log(Surface Brûlée) en Fonction de la Température")
ax1.grid(True, linestyle="--", alpha=0.5)

# 📂 Sauvegarde du graphique
plt.savefig(f"{save_dir}/surface_brulee_et_surfacelog_vs_hr.png")
plt.savefig(f"{save_dir}/surface_brulee_etsurfacelog_vs_hr.pdf")
plt.close()

# Visualiser la surface brûlée en fonction du vent
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_area_non_0, x='wind', y='area', alpha=0.6, edgecolor=None, color="royalblue")
plt.title("Surface Brûlée en Fonction du vent")
plt.xlabel("vent (km/h)")
plt.ylabel("Surface Brûlée (ha)")
plt.grid(True)
plt.tight_layout()

# 📂 Sauvegarde du graphique
plt.savefig(f"{save_dir}/surface_brulee_vs_vent.png")
plt.savefig(f"{save_dir}/surface_brulee_vs_vent.pdf")
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

# 📂 Sauvegarde du graphique
plt.savefig(f"{save_dir}/surface_brulee_et_surfacelog_vs_vent.png")
plt.savefig(f"{save_dir}/surface_brulee_et_surfacelog_vs_vent.pdf")
plt.close()

# Visualisation d'autres pairplots

# 📌 Fonction de sauvegarde pour les heatmaps et pairplots
def save_figure(fig, file_name, save_dir):
    """Sauvegarde les figures sous format PNG et PDF."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Créer le répertoire s'il n'existe pas

    fig.savefig(os.path.join(save_dir, f"{file_name}.png"), bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, f"{file_name}.pdf"), bbox_inches='tight')
    plt.close(fig)

# Liste des colonnes numériques à analyser
# 🟢 Définition des groupes de colonnes
colonnes_normales = ['X', 'Y', 'BUI', 'temp']
colonnes_asymetriques = ['FFMC', 'DMC', 'DC', 'ISI', 'RH', 'wind', 'rain']

# Affichage du pairplot pour le cas où la surface brûlée est non nulle
pairplot_burned = sns.pairplot(df_area_non_0[colonnes_normales+ colonnes_asymetriques + ['log_area']])
plt.suptitle("Pairplot - Surface brûlée non nulle", y=1.02)
save_figure(pairplot_burned.fig, "pairplot_burned", save_dir)

# Affichage du pairplot pour le cas où la surface brûlée est nulle
pairplot_no_burn = sns.pairplot(df_area_0[colonnes_normales + colonnes_asymetriques])
plt.suptitle("Pairplot - Surface brûlée nulle", y=1.02)
save_figure(pairplot_burned.fig, "pairplot_no_burn", save_dir)

print("Les pairplots ont été générés et sauvegardés.")

# Tracé des valeurs min, max et moyennes du FWI en fonction du mois

df_grouped = df_cleaned.groupby('month')['FWI'].agg(['max', 'min', 'mean'])

# Assurez-vous que les mois sont ordonnés
df_grouped = df_grouped.sort_index()

# Création de la figure et des axes
plt.figure(figsize=(10, 6))

# Tracer les trois courbes
plt.plot(df_grouped.index, df_grouped['max'], label='FWI Max', color='red', marker='o')
plt.plot(df_grouped.index, df_grouped['min'], label='FWI Min', color='blue', marker='o')
plt.plot(df_grouped.index, df_grouped['mean'], label='FWI Moyen', color='green', marker='o')

# Ajouter des labels, un titre et une légende
plt.xlabel('Mois')
plt.ylabel('FWI')
plt.title('FWI Max, Min et Moyen en fonction du mois')

# Réordonner les mois de 1 à 12 sur l'axe x
mois_ordre = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(ticks=df_grouped.index, labels=mois_ordre)

# Ajouter la légende
plt.legend()

# Sauvegarder le graphique sous .png et .pdf
plt.savefig(f'{save_dir}/fwi_courbes_en_fonction_mois_calendaire.png')
plt.savefig(f'{save_dir}/fwi_courbes_en_fonction_mois_calendaire.pdf')
plt.close()

# Analyses multivariées

# Définir un répertoire et le créer si non encore existant
save_dir = "analyses_multivariées"
os.makedirs(save_dir, exist_ok=True)

# 🟢 Définition des groupes de colonnes
colonnes_normales = ['X', 'Y', 'BUI', 'temp']
colonnes_asymetriques = ['FFMC', 'DMC', 'DC', 'ISI', 'RH', 'wind', 'rain']

# 📊 Matrices de corrélation
correlation_pearson = df_cleaned[['log_area']+colonnes_normales].corr(method='pearson')  # (Normales vs Normales)
correlation_spearman_asym = df_cleaned[['log_area']+colonnes_asymetriques].corr(method='spearman')  # (Asymétriques vs Asymétriques)
correlation_spearman_mixed = df_cleaned[['log_area']+colonnes_normales + colonnes_asymetriques].corr(method='spearman')  # (Tout en Spearman)

# 📌 Affichage des matrices
print("\n🔹 Matrice de Corrélation Pearson (Colonnes Normales) :\n", correlation_pearson)
print("\n🔹 Matrice de Corrélation Spearman (Colonnes Asymétriques) :\n", correlation_spearman_asym)
print("\n🔹 Matrice de Corrélation Spearman (Mélange Normales & Asymétriques) :\n", correlation_spearman_mixed)

# 🔥 Création des heatmaps
fig, axes = plt.subplots(1, 3, figsize=(25, 7))

# 🟢 Heatmap Pearson (Normales vs Normales)
sns.heatmap(correlation_pearson, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[0])
axes[0].set_title("🔹 Matrice de Corrélation de Pearson (Colonnes Normales)")

# 🔴 Heatmap Spearman (Asymétriques vs Asymétriques)
sns.heatmap(correlation_spearman_asym, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[1])
axes[1].set_title("🔹 Matrice de Corrélation de Spearman (Colonnes Asymétriques)")

# 🔄 Heatmap Spearman (Mélange Normales et Asymétriques)
sns.heatmap(correlation_spearman_mixed, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[2])
axes[2].set_title("🔹 Matrice de Corrélation de Spearman (Tout)")

# 📂 Sauvegarde des figures
plt.tight_layout()
plt.savefig(f"{save_dir}/correlation_heatmaps.png")
plt.savefig(f"{save_dir}/correlation_heatmaps.pdf")
plt.close()

# Poursuite analyses multivariées en distinguant les cas surface brûlée nulle et les cas surface brulée non nulle

# Définir un répertoire et le créer si non encore existant
save_dir = "analyses_multivariees_surface_brulee_0_et_non_0"
os.makedirs(save_dir, exist_ok=True)

# Statistiques descriptives pour les sous-groupes
print("Statistiques descriptives - Surface brûlée non nulle:")
print(df_area_non_0.describe())

print("\nStatistiques descriptives - Surface brûlée nulle:")
print(df_area_0.describe())

def save_heatmap(heatmap, file_name, save_dir):
    """Sauvegarde les heatmaps sous format PNG et PDF."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Créer le répertoire s'il n'existe pas

    plt.savefig(os.path.join(save_dir, f"{file_name}.png"), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f"{file_name}.pdf"), bbox_inches='tight')
    plt.close()

# Liste des colonnes numériques à analyser
variables_numeriques = ['X', 'Y', 'BUI', 'temp', 'FFMC', 'DMC', 'DC', 'ISI', 'RH', 'wind', 'rain']

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

# Création du répertoire de sauvegarde
#save_dir = "analyses_multivariees_surface_brulee_0_et_non_0"

# 📌 Corrélation Pearson pour les colonnes normales
if colonnes_normales_brule:
    pearson_corr_burned = df_area_non_0[['log_area']+colonnes_normales_brule].corr(method='pearson')
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

# 📌 Corrélation Spearman pour les colonnes asymétriques
if colonnes_asymetriques_brule:
    spearman_corr_burned = df_area_non_0[['log_area']+colonnes_asymetriques_brule].corr(method='spearman')
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

# Statistiques inférentielles pour étudier l'influence des colonnes sur la surface brûlée

# Définir un répertoire et le créer si non encore existant
save_dir = "statistiques_inférentielles"
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
variables = ['X', 'Y', 'BUI', 'temp', 'FFMC', 'DMC', 'DC', 'ISI', 'RH', 'wind']

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

    # Sauvegarder l'histogramme dans le PDF
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
#save_dir = "statistiques_inférentielles"
#os.makedirs(save_dir, exist_ok=True)

# 🔹 Convertir X et Y en catégories pour le test du Chi²
df_area_non_0["X_cat"] = df_area_non_0["X"].astype("category")
df_area_non_0["Y_cat"] = df_area_non_0["Y"].astype("category")

# 🔹 Effectuer les tests du Chi² et stocker les résultats
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

# 🔹 Création du PDF
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

# maintenant regarder opportunité transformation log area, puis faire autres test statistiques inférentielles avec Python/R
# puis faire si possible et autres analyses poussées pour arriver à établir relations avec paramètres météo
# regarder les tests possibles pour savoir si relation non linéaire et l'établir
#puis envisager de lancer des modèles types PCA, clustering, régression logistique, random forest et autres modèles avec R, Python Scikit learn et autres
# ensuite regarder s'il faut enlever des outliers
# regarder aussi établissement d'une échelle de risque avec les relations trouvées et FWI
# puis faire l'injection d'une base de données propre avec données clés et trouvées et propres vers postgresql

# ANOVA et test de Kruskal-Wallis pour comparer surfaces brulées selon catégories des différentes colonnes

# Liste des variables continues (excluant area et log_area)
variables_continues = ['X', 'Y', 'BUI', 'temp', 'FFMC', 'DMC', 'DC', 'ISI', 'RH', 'wind']

# Dictionnaire pour stocker les résultats
test_results = []

# 📌 Catégorisation des variables continues en classes
df_area_0_cat = df_area_0.copy()
df_area_non_0_cat = df_area_non_0.copy()

for var in variables_continues:
    df_area_0_cat[var + "_cat"] = pd.qcut(df_area_0[var], q=3, labels=["Bas", "Moyen", "Élevé"])
    df_area_non_0_cat[var + "_cat"] = pd.qcut(df_area_non_0[var], q=3, labels=["Bas", "Moyen", "Élevé"])

# 📌 Séparation des variables normales et asymétriques via le test de Shapiro-Wilk
variables_normales = []
variables_asymetriques = []

for var in variables_continues:
    p_value_non_0 = stats.shapiro(df_area_non_0[var])[1] if len(df_area_non_0[var]) > 3 else 1
    p_value_0 = stats.shapiro(df_area_0[var])[1] if len(df_area_0[var]) > 3 else 1

    if p_value_non_0 > 0.05 and p_value_0 > 0.05:
        variables_normales.append(var)
    else:
        variables_asymetriques.append(var)

# 📌 ANOVA pour les variables normales
for var in variables_normales:
    f_stat, p_value = stats.f_oneway(df_area_non_0[var], df_area_0[var])
    result = f"ANOVA pour {var} : F = {f_stat:.3f}, p-value = {p_value:.3f}"
    result += "\n-> Différence significative entre les groupes." if p_value < 0.05 else "\n-> Aucune différence significative."
    test_results.append(result)

# 📌 Kruskal-Wallis pour les variables asymétriques
for var in variables_asymetriques:
    h_stat, p_value = stats.kruskal(df_area_non_0[var], df_area_0[var])
    result = f"Kruskal-Wallis pour {var} : H = {h_stat:.3f}, p-value = {p_value:.3f}"
    result += "\n-> Différence significative entre les groupes." if p_value < 0.05 else "\n-> Aucune différence significative."
    test_results.append(result)

# 📌 Sauvegarde des résultats dans un PDF
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

# Analyses statistiques complémentaires avec R

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

# 1. Test de normalité (Shapiro-Wilk et Kolmogorov-Smirnov) sur FWI
print("\nTest de normalité en R :")
shapiro_test = stats.shapiro_test(df_r.rx2("FWI"))
ks_test = stats.ks_test(df_r.rx2("FWI"), "pnorm", mean=df_cleaned['FWI'].mean(), sd=df_cleaned['FWI'].std())
print(f"Shapiro-Wilk p-value : {shapiro_test[1]}")
print(f"Kolmogorov-Smirnov p-value : {ks_test[1]}")

# 2. ANOVA (Analyse de la variance) pour tester les différences entre les saisons
print("\nAnalyse de la variance (ANOVA) en R :")
anova_model = stats.aov(r('FWI ~ season'), data=df_r)
print(base.summary(anova_model))

# 3. Régression non linéaire (polynomiale) entre FWI et Température
print("\nRégression polynomiale en R :")
poly_model = stats.lm(r('FWI ~ poly(temp, 2)'), data=df_r)
print(base.summary(poly_model))

# 4. Régression multiple entre Température et Humidité relative (avec la surface brûlée non nulle)
df_r_area_non_0 = pandas2ri.py2rpy(df_area_non_0)

# temp et rh sont les variables explicatives
print("\nRégression multiple en R : Température et Humidité relative sur surface brûlée non nulle")

# Passer l'objet à R pour la transformation et le modèle
#r('df_r_area_non_0 <- ' + str(df_r_area_non_0))  # Importer correctement le DataFrame dans R

# Transformer 'RH' en log(RH) car distribution asymétrique
#r('df_r_area_non_0$log_RH <- log(df_r_area_non_0$RH)')

# Créer le modèle de régression multiple
lm_model_rh_temp = r.lm('log_area ~ temp + RH', data=df_r_area_non_0)

# Afficher le résumé du modèle
#print(r.summary(lm_model_rh_temp))

# Résumé du modèle
print(base.summary(lm_model_rh_temp))

# Régression multiple entre BUI et Température comme variables explicatives pour prédire la surface brûlée
print("\nRégression multiple en R : BUI et Température sur surface brûlée")

# Création du modèle de régression multiple
lm_model_temp_bui = r.lm('log_area ~ temp + BUI', data=df_r_area_non_0)

# Résumé du modèle
print(base.summary(lm_model_temp_bui))


""""# 4. Clustering hiérarchique pour regrouper les jours selon leurs FWI et conditions
print("\nClustering hiérarchique en R :")
clustering_model = cluster.hclust(stats.dist(df_r.rx2("FWI")), method="ward.D2")
r("plot")(clustering_model, main="Clustering hiérarchique des FWI", sub="", xlab="Jours")

# 5. Résultat du clustering
cluster_assignments = r.cutree(clustering_model, k=3)
print("Cluster assignments :", cluster_assignments)"""

# 4 Création et Alimentation de la Base de données traitée en utilisant PostGreSQL depuis Python

# Rappel des informations sur le dataframe traité à l'étape 2
print("Rappel des informations sur le dataframe df_cleaned")
print(df_cleaned.info())

# Convertir la colonne 'season' en chaînes de caractères
df_cleaned['day'] = df_cleaned['day'].astype(str)
df_cleaned['month'] = df_cleaned['month'].astype(str)
df_cleaned['season'] = df_cleaned['season'].astype(str)
df_cleaned['danger_level'] = df_cleaned['danger_level'].astype(str)
df_cleaned['level_description'] = df_cleaned['level_description'].astype(str)

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
        client_encoding='UTF8'  # Assurez-vous que l'encodage est UTF-8
    )


# Créer la table dans la base de données
def create_table(cur):
    table_creation_query = """
    CREATE TABLE IF NOT EXISTS data_table_4 (
        X INTEGER,
        Y INTEGER,
        month VARCHAR,
        day VARCHAR,
        FFMC FLOAT,
        DMC FLOAT,
        DC FLOAT,
        ISI FLOAT,
        temp FLOAT,
        RH INTEGER,
        wind FLOAT,
        rain FLOAT,
        area FLOAT,
        season VARCHAR,
        BUI FLOAT,
        FWI FLOAT,
        danger_level VARCHAR,
        level_description VARCHAR,
        log_area FLOAT
    );
    """
    cur.execute(table_creation_query)

#PRIMARY KEY (X, Y, month, day)

# Insérer les données du DataFrame dans la table
def insert_data_from_dataframe(cur, df):
    # Convertir le DataFrame en une liste de tuples
    data_tuples = [tuple(row) for row in df[['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI',
                                             'temp', 'RH', 'wind', 'rain', 'area', 'season',
                                             'BUI', 'FWI', 'danger_level', 'level_description',
                                             'log_area']].values]

    insert_query = """
    INSERT INTO data_table_4 (
        X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain, area,
        season, BUI, FWI, danger_level, level_description, log_area
    ) VALUES %s;
    """

    # Insertion des données en une seule requête
    execute_values(cur, insert_query, data_tuples)


# Paramètres de connexion
database_name = "viken_db_4"  # Nom de votre base de données
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

except Exception as e:
    print(f"Erreur d'exécution : {e}")

finally:
    # Fermer la connexion et le curseur
    if cur:
        cur.close()
    if conn:
        conn.close()
