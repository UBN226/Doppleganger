import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import time

import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import scipy.stats as stats

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================
st.set_page_config(
    page_title="Doppleganger Finder",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS PERSONNALISÉ - THÈME MODERNE
# ============================================
st.markdown("""
<style>
    /* Variables CSS */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #22c55e;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
    }
    
    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }
    
    /* Cards pour les résultats */
    .result-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 16px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        transition: all 0.3s ease;
        margin-bottom: 12px;
    }
    
    .result-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.6);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.15);
    }
    
    /* Badge de confiance */
    .confidence-badge {
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-top: 8px;
    }
    
    .confidence-high {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
    }
    
    /* Titre principal */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Stats card */
    .stat-card {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling - Compatible Streamlit */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }
    
    /* Sidebar text and labels */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p {
        color: #f1f5f9 !important;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }
    
    /* Sidebar metrics */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #6366f1 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    
    /* Sidebar slider */
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: #6366f1 !important;
    }
    
    /* File uploader */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        padding: 10px;
        border: 1px dashed rgba(99, 102, 241, 0.3);
    }
    
    /* Camera input */
    [data-testid="stSidebar"] [data-testid="stCameraInput"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Amélioration des images */
    .stImage > img {
        border-radius: 12px;
        border: 2px solid rgba(99, 102, 241, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# CHARGEMENT DU MODÈLE
# ============================================
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    model = InceptionResnetV1(pretrained="vggface2").to(device).eval()
    return mtcnn, model, device

mtcnn, model, device = load_model()


# ============================================
# FONCTIONS UTILITAIRES
# ============================================
def process_query_image(pil_img):
    """Détecte et extrait le visage d'une image."""
    face = mtcnn(pil_img)
    if face is None:
        return None, None
    
    # Tensor → image PIL (pour affichage)
    face_img = face.permute(1, 2, 0)
    face_img = (face_img + 1) / 2
    face_img = (face_img * 255).byte().cpu().numpy()
    face_pil = Image.fromarray(face_img)
    
    return face, face_pil


def extract_embedding_from_face(face_tensor):
    """Extrait l'embedding d'un visage détecté."""
    x = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x).cpu().numpy().flatten()
    return emb / np.linalg.norm(emb)


def get_confidence_level(score):
    """Retourne le niveau de confiance et la classe CSS."""
    if score >= 0.7:
        return "Très ressemblant", "confidence-high", "🟢"
    elif score >= 0.5:
        return "Ressemblance modérée", "confidence-medium", "🟡"
    else:
        return "Ressemblance faible", "confidence-low", "🔴"


def compute_percentile(score, all_scores):
    """Calcule le percentile du score par rapport à tous les scores."""
    return stats.percentileofscore(all_scores, score)


# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
@st.cache_data
def load_data():
    base = Path("../data/processed")
    embeddings = np.load(base / "embeddings.npy")
    meta = pd.read_csv(base / "meta.csv")
    return embeddings, meta

embeddings, meta = load_data()


# ============================================
# INTERFACE PRINCIPALE
# ============================================

# Titre
st.markdown('<h1 class="main-title">🧬 Doppleganger Finder</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Trouvez vos sosies grâce au Deep Learning et à la similarité cosinus</p>', unsafe_allow_html=True)

# ============================================
# SIDEBAR - PARAMÈTRES
# ============================================
st.sidebar.markdown("## ⚙️ Paramètres")

# Choix de la méthode d'acquisition
input_method = st.sidebar.radio(
    "📷 Source de l'image",
    ["📤 Importer une image", "📸 Prendre une photo"],
    index=0
)

st.sidebar.markdown("---")

top_k = st.sidebar.slider("🔢 Nombre de sosies à afficher", 3, 10, 5)

source_filter = st.sidebar.radio(
    "🗂️ Filtrer par source",
    ["Toutes", "FairFace", "Photos personnelles"]
)

st.sidebar.markdown("---")

# Statistiques de la base
st.sidebar.markdown("### 📊 Base de données")
total_images = len(meta)
fairface_count = len(meta[meta["source"] == "fairface"])
personal_count = len(meta[meta["source"] == "our_faces"])

col1, col2 = st.sidebar.columns(2)
col1.metric("Total", f"{total_images:,}")
col2.metric("FairFace", f"{fairface_count:,}")


# ============================================
# ACQUISITION DE L'IMAGE
# ============================================
query_img = None

if input_method == "📤 Importer une image":
    uploaded_file = st.sidebar.file_uploader(
        "Choisir une image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        query_img = Image.open(uploaded_file).convert("RGB")

elif input_method == "📸 Prendre une photo":
    camera_photo = st.sidebar.camera_input("Prenez une photo")
    if camera_photo is not None:
        query_img = Image.open(camera_photo).convert("RGB")


# ============================================
# TRAITEMENT ET AFFICHAGE DES RÉSULTATS
# ============================================
if query_img is not None:
    
    # Détection du visage
    with st.spinner("🔍 Détection du visage..."):
        face_tensor, face_crop = process_query_image(query_img)
    
    if face_tensor is None:
        st.error("❌ Aucun visage détecté dans l'image. Veuillez réessayer avec une autre photo.")
        st.stop()
    
    # Affichage image originale et visage détecté
    st.markdown('<div class="section-header">📸 Votre Image</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(query_img, caption="Image originale", use_container_width=True)
    with col2:
        st.image(face_crop, caption="Visage détecté", use_container_width=True)
    
    # Extraction de l'embedding et calcul des similarités
    with st.spinner("🧠 Analyse en cours..."):
        query_emb = extract_embedding_from_face(face_tensor)
        sims = cosine_similarity(query_emb.reshape(1, -1), embeddings)[0]
        meta_copy = meta.copy()
        meta_copy["similarity"] = sims
    
    # Filtrage par source
    if source_filter == "FairFace":
        results = meta_copy[meta_copy["source"] == "fairface"]
    elif source_filter == "Photos personnelles":
        results = meta_copy[meta_copy["source"] == "our_faces"]
    else:
        results = meta_copy
    
    topk = results.sort_values("similarity", ascending=False).head(top_k)
    
    # ============================================
    # RÉSULTATS - SOSIES TROUVÉS
    # ============================================
    st.markdown('<div class="section-header">🎭 Vos Sosies</div>', unsafe_allow_html=True)
    
    cols = st.columns(top_k)
    
    for col, (_, row) in zip(cols, topk.iterrows()):
        with col:
            img = Image.open(row["path"]).convert("RGB")
            st.image(img, use_container_width=True)
            
            score = row["similarity"]
            label, css_class, emoji = get_confidence_level(score)
            
            st.markdown(f"""
                <div class="result-card">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #f1f5f9;">
                        {score:.1%}
                    </div>
                    <div class="confidence-badge {css_class}">
                        {emoji} {label}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # ============================================
    # STATISTIQUES ENRICHIES
    # ============================================
    st.markdown('<div class="section-header">📈 Analyse Détaillée</div>', unsafe_allow_html=True)
    
    best_score = topk["similarity"].max()
    mean_score = topk["similarity"].mean()
    std_score = topk["similarity"].std()
    percentile = compute_percentile(best_score, sims)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        label, _, emoji = get_confidence_level(best_score)
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{best_score:.1%}</div>
                <div class="stat-label">{emoji} Meilleur score</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{mean_score:.1%}</div>
                <div class="stat-label">📊 Score moyen (Top-{top_k})</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{std_score:.3f}</div>
                <div class="stat-label">📉 Écart type</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">Top {100 - percentile:.0f}%</div>
                <div class="stat-label">🏆 Percentile</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Graphiques
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**📊 Distribution des scores (Top-K)**")
        
        fig1, ax1 = plt.subplots(figsize=(8, 4), facecolor='#0f172a')
        ax1.set_facecolor('#1e293b')
        
        bars = ax1.barh(
            range(len(topk)),
            topk["similarity"].values[::-1],
            color=['#22c55e' if s >= 0.7 else '#f59e0b' if s >= 0.5 else '#ef4444' 
                   for s in topk["similarity"].values[::-1]],
            edgecolor='white',
            linewidth=0.5
        )
        
        ax1.set_yticks(range(len(topk)))
        ax1.set_yticklabels([f"Sosie #{i+1}" for i in range(len(topk))][::-1], color='white')
        ax1.set_xlabel("Score de similarité", color='white')
        ax1.tick_params(colors='white')
        ax1.set_xlim(0, 1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color('white')
        ax1.spines['left'].set_color('white')
        
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col_chart2:
        st.markdown("**🎯 Position dans la base de données**")
        
        fig2, ax2 = plt.subplots(figsize=(8, 4), facecolor='#0f172a')
        ax2.set_facecolor('#1e293b')
        
        # Histogramme de tous les scores
        ax2.hist(sims, bins=50, color='#6366f1', alpha=0.7, edgecolor='white', linewidth=0.3)
        
        # Ligne verticale pour le meilleur score
        ax2.axvline(best_score, color='#22c55e', linestyle='--', linewidth=2, 
                    label=f'Meilleur match: {best_score:.2f}')
        
        ax2.set_xlabel("Score de similarité", color='white')
        ax2.set_ylabel("Nombre d'images", color='white')
        ax2.tick_params(colors='white')
        ax2.legend(facecolor='#1e293b', edgecolor='white', labelcolor='white')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_color('white')
        ax2.spines['left'].set_color('white')
        
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Interprétation
    st.markdown("---")
    
    if best_score >= 0.7:
        st.success(f"""
            🎉 **Excellent résultat !** Votre meilleur sosie a un score de **{best_score:.1%}**, 
            ce qui indique une ressemblance très marquée. Vous vous situez dans le **top {100 - percentile:.0f}%** 
            des correspondances de la base.
        """)
    elif best_score >= 0.5:
        st.warning(f"""
            👍 **Bonne correspondance !** Votre meilleur sosie a un score de **{best_score:.1%}**, 
            ce qui indique une ressemblance modérée. L'écart type de **{std_score:.3f}** montre que 
            les sosies trouvés ont des scores {'similaires' if std_score < 0.05 else 'variés'}.
        """)
    else:
        st.info(f"""
            🔍 **Résultat intéressant !** Même si le score de **{best_score:.1%}** est modeste, 
            cela peut indiquer que votre visage a des caractéristiques uniques. 
            Essayez avec une photo de meilleure qualité ou un autre angle.
        """)

else:
    # État initial - Pas d'image
    st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📷</div>
            <h3 style="color: #f1f5f9; margin-bottom: 0.5rem;">Commencez votre recherche</h3>
            <p style="color: #94a3b8;">
                Importez une photo ou prenez un selfie depuis la barre latérale pour trouver vos sosies.
            </p>
        </div>
    """, unsafe_allow_html=True)
