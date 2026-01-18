# 🧬 Doppleganger Finder

Une application de **Deep Learning** qui trouve vos sosies dans une base de données de visages en utilisant la reconnaissance faciale et la similarité cosinus.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)

---

## 📋 Table des Matières

- [Présentation](#-présentation)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture Technique](#-architecture-technique)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Interprétation des Statistiques](#-interprétation-des-statistiques)
- [Structure du Projet](#-structure-du-projet)
- [Datasets](#-datasets)

---

## 🎯 Présentation

**Doppleganger Finder** est un projet de Deep Learning qui permet de trouver les personnes qui vous ressemblent le plus dans une base de données d'images. L'application utilise un modèle CNN pré-entraîné pour extraire des caractéristiques faciales (embeddings) et calcule la similarité cosinus pour identifier les visages les plus proches.

### Principe de fonctionnement

1. **Détection du visage** : MTCNN détecte et extrait le visage de l'image
2. **Extraction des features** : InceptionResnetV1 (VGGFace2) génère un vecteur de 512 dimensions
3. **Calcul de similarité** : Comparaison cosinus avec tous les embeddings de la base
4. **Classement** : Affichage des Top-K visages les plus similaires

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| 📤 **Import d'image** | Chargez une photo depuis votre appareil |
| 📸 **Capture webcam** | Prenez une photo directement avec votre caméra |
| 🎯 **Détection faciale** | MTCNN pour une détection précise des visages |
| 🧠 **Embeddings CNN** | InceptionResnetV1 pré-entraîné sur VGGFace2 |
| 📊 **Statistiques avancées** | Score de confiance, percentile, distribution |
| 🎨 **Interface moderne** | Design dark theme avec effets visuels |
| 🔍 **Filtres** | Par source (FairFace, photos personnelles) |

---

## 🏗 Architecture Technique

```
┌─────────────────────────────────────────────────────────────┐
│                    Image d'entrée                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    MTCNN                                    │
│              (Détection + Alignement)                       │
│                  Sortie: 160x160                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              InceptionResnetV1 (VGGFace2)                   │
│                                                             │
│   Conv → Inception Blocks → AvgPool → FC → L2 Normalize     │
│                                                             │
│                  Sortie: Vecteur 512D                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               Similarité Cosinus                            │
│                                                             │
│         sim(A,B) = (A · B) / (||A|| × ||B||)                │
│                                                             │
│              Comparaison avec 10,000+ embeddings            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Top-K Résultats                            │
│           (triés par score de similarité)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Webcam (optionnel, pour la capture photo)

### Étapes d'installation

```bash
# 1. Cloner le projet
git clone <repository-url>
cd Doppleganger

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
cd app
streamlit run app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

---

## 📖 Utilisation

### 1. Charger une image

Deux options disponibles dans la barre latérale :
- **Importer une image** : Sélectionnez un fichier JPG/PNG depuis votre ordinateur
- **Prendre une photo** : Utilisez votre webcam pour capturer un selfie

### 2. Configurer les paramètres

- **Nombre de sosies** : Choisissez entre 3 et 10 résultats
- **Filtrer par source** : 
  - *Toutes* : Cherche dans toute la base
  - *FairFace* : Uniquement le dataset FairFace
  - *Photos personnelles* : Uniquement vos photos ajoutées

### 3. Analyser les résultats

L'application affiche :
- Votre image originale et le visage détecté
- Les sosies trouvés avec leur score de similarité
- Des statistiques détaillées et visualisations

---

## 📊 Interprétation des Statistiques

### Métriques Principales

| Métrique | Signification | Interprétation |
|----------|---------------|----------------|
| **Meilleur score** | Score de similarité du sosie #1 | Plus c'est proche de 100%, plus la ressemblance est forte |
| **Score moyen (Top-K)** | Moyenne des scores des K sosies | Indique la qualité globale des correspondances |
| **Écart type** | Dispersion des scores | Faible = sosies similaires entre eux, Élevé = grande diversité |
| **Percentile** | Position dans la base | "Top 5%" = votre match est meilleur que 95% des autres |

### Niveaux de Confiance

| Score | Niveau | Badge | Signification |
|-------|--------|-------|---------------|
| ≥ 70% | 🟢 Très ressemblant | Vert | Ressemblance marquée, traits faciaux très proches |
| 50-70% | 🟡 Ressemblance modérée | Orange | Certains traits communs, ressemblance partielle |
| < 50% | 🔴 Ressemblance faible | Rouge | Peu de traits communs, correspondance limitée |

### Graphiques

#### Distribution des scores (Top-K)
- **Barres horizontales** : Score de chaque sosie
- **Couleur** : Vert (≥70%), Orange (50-70%), Rouge (<50%)
- **À observer** : Des barres de longueurs similaires indiquent plusieurs bons matchs

#### Position dans la base de données
- **Histogramme** : Distribution de tous les scores de similarité
- **Ligne verte verticale** : Position de votre meilleur match
- **À observer** : Plus la ligne est à droite, meilleur est votre match par rapport à la base

### Exemple d'interprétation

> **Résultat** : Meilleur score 99.7%, Score moyen 83.5%, Écart type 0.093, Top 0%

**Analyse** :
- ✅ **99.7%** : Ressemblance exceptionnelle avec le sosie #1
- ✅ **83.5%** : Tous les sosies ont des scores élevés
- ✅ **0.093** : Très faible → Les 5 sosies se ressemblent beaucoup entre eux
- ✅ **Top 0%** : Meilleur match possible dans toute la base (meilleur que 100% des autres visages)

---

## 📁 Structure du Projet

```
dl_project/
├── app/
│   └── app.py              # Application Streamlit
├── data/
│   ├── raw/
│   │   ├── fairface/       # Images brutes FairFace
│   │   └── our_faces/      # Photos personnelles brutes
│   └── processed/
│       ├── embeddings.npy           # Embeddings combinés
│       ├── embeddings_fairface.npy  # Embeddings FairFace
│       ├── embeddings_our_faces.npy # Embeddings photos perso
│       ├── meta.csv                 # Métadonnées (chemin, source)
│       ├── fairface_faces/          # Visages extraits FairFace
│       └── our_faces/               # Visages extraits perso
├── notebooks/
│   ├── Explore_FairFace.ipynb       # Exploration du dataset
│   ├── pretraitements.ipynb         # Prétraitement des images
│   ├── embedding.ipynb              # Génération des embeddings
│   └── fusion_emb_similarite.ipynb  # Fusion et similarité
├── description/
│   └── Projets.md          # Description du sujet
├── requirements.txt        # Dépendances Python
└── README.md               # Ce fichier
```

---

## 📚 Datasets

### FairFace
- **Source** : [FairFace Dataset](https://github.com/joojs/fairface)
- **Taille** : ~10,500 images
- **Description** : Dataset équilibré en termes d'âge, genre et origine ethnique

### Photos Personnelles
- **Taille** : Variable (ajoutées par l'utilisateur)
- **Format** : JPG, PNG
- **Emplacement** : `data/raw/our_faces/`

---

## 🔧 Technologies Utilisées

| Technologie | Version | Usage |
|-------------|---------|-------|
| PyTorch | 2.0+ | Framework Deep Learning |
| facenet-pytorch | 2.5+ | MTCNN + InceptionResnetV1 |
| Streamlit | 1.28+ | Interface utilisateur web |
| scikit-learn | 1.0+ | Similarité cosinus |
| NumPy | 1.20+ | Manipulation d'arrays |
| Pandas | 2.0+ | Gestion des métadonnées |
| Matplotlib | 3.5+ | Visualisations |
| Pillow | 9.0+ | Traitement d'images |
| SciPy | 1.10+ | Calculs statistiques |

---

## 👥 Auteurs

Projet réalisé dans le cadre du cours **Réseaux de Neurones & Deep Learning** - Master 2 IDSI.

---

## 📄 Licence

Ce projet est à but éducatif.
