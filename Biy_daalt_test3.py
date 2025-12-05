import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Optional, List
import torch
from torchvision import models, transforms
# Euclidean Distance-ийг импорт хийнэ
from sklearn.metrics.pairwise import euclidean_distances 

IMAGE_DIR = Path("C:/Users/Lenovo/Documents/python/F.ITM324/src/famous_art_images")
DOWNLOADED_CSV_FILE = Path("C:/Users/Lenovo/Documents/python/F.ITM324/src/art.csv")

TOP_K_DEFAULT = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Metadata-г Унших ба Боловсруулах ---
try:
    metadata_df_base = pd.read_csv(DOWNLOADED_CSV_FILE)
    metadata_df_base = metadata_df_base[['id', 'name', 'title', 'creator', 'description']]
    st.sidebar.success(f"Metadata амжилттай ачааллаа. Нийт {len(metadata_df_base)} мөр байна.")
except FileNotFoundError:
    st.error(f"Алдаа: Metadata файл олдсонгүй! Замыг шалгана уу: {DOWNLOADED_CSV_FILE}")
    st.stop()

# --- Модел, трансформ (ResNet50-ийг ашиглана) ---
@st.cache_resource
def load_model_and_extractor():
    base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1])
    feature_extractor.to(DEVICE)
    feature_extractor.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return feature_extractor, preprocess

feature_extractor, preprocess = load_model_and_extractor()

# --- Зургийн feature гаргах (Өөрчлөлтгүй) ---
def get_image_features(image_file, feature_extractor, preprocess, device=DEVICE) -> Optional[np.ndarray]:
    try:
        img = Image.open(image_file).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = feature_extractor(input_tensor)
            feats = feats.view(feats.size(0), -1)
            feat_np = feats.cpu().numpy().reshape(-1)
            norm = np.linalg.norm(feat_np)
            if norm > 0:
                feat_np = feat_np / norm # Векторыг нормчилж байгаа
            return feat_np
    except Exception as e:
        st.error(f"Feature гаргах үед алдаа: {e}")
        return None

# --- Feature индекс үүсгэх (Өөрчлөлтгүй) ---
@st.cache_data
def build_feature_index_cached(image_dir: Path, metadata_df: pd.DataFrame,
                               _feature_extractor, _preprocess):
    feature_index: Dict[str, np.ndarray] = {}
    ok_rows: List[int] = []

    for idx, row in metadata_df.iterrows():
        img_path = image_dir / row['name']
        if not img_path.exists():
            continue
        feat = get_image_features(img_path, _feature_extractor, _preprocess)
        if feat is not None:
            feature_index[row['id']] = feat
            ok_rows.append(idx)

    filtered_metadata = metadata_df.loc[ok_rows].reset_index(drop=True)
    return feature_index, filtered_metadata

# ТУРШИЛТ 3: Euclidean Distance-ээр Хайлт Хийх
def search_euclidean_art(query_feat: np.ndarray,
                         feature_index: Dict[str, np.ndarray],
                         metadata_df: pd.DataFrame,
                         top_k: int = TOP_K_DEFAULT):
    if len(feature_index) == 0:
        return "Индекс хоосон байна.", None
        
    indexed_ids = list(feature_index.keys())
    indexed_vectors = np.vstack([feature_index[_id] for _id in indexed_ids])
    
    # Euclidean Distance-ийг тооцоолно (Зай хэмжинэ)
    distances = euclidean_distances(query_feat.reshape(1,-1), indexed_vectors).flatten()
    
    # Эрэмбэлэлт: Хамгийн БАГА зайтайг сонгоно (өсөх дараалал)
    ranked_idx = np.argsort(distances)[:top_k]
    
    results = []
    for rank, ridx in enumerate(ranked_idx, start=1):
        doc_id = indexed_ids[ridx]
        # Зайг оноо болгон харуулж байна. (Бага байх тусам сайн)
        distance_score = float(distances[ridx]) 
        info = metadata_df[metadata_df['id']==doc_id].iloc[0]
        results.append({
            'rank': rank,
            'score': distance_score, # Энэ бол зай (Distance)
            'title': info['title'],
            'creator': info['creator'],
            'description': info['description'],
            'name': info['name']
        })
    return "Хайлт амжилттай.", results

# --- Streamlit интерфейс ---
st.set_page_config(page_title="Алдартай Зургийн Хайлт", layout="wide")
st.title(" Алдартай Зургийг Таних ба Тайлбарлах (Туршилт 3: Euclidean Distance)")

# 1️⃣ Индекс үүсгэх
feature_index, filtered_meta = build_feature_index_cached(
    IMAGE_DIR, metadata_df_base, feature_extractor, preprocess
)

# 2️⃣ Query зураг оруулах
st.header("1. Зураг ачаалж хайх")
uploaded_file = st.file_uploader("Зургийн файл сонгоно уу (JPG, PNG)", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption=uploaded_file.name, width=200)
    query_feat = get_image_features(uploaded_file, feature_extractor, preprocess)
    
    if query_feat is not None:
        #  Euclidean Distance функцээр хайлт хийнэ
        status, results = search_euclidean_art(query_feat, feature_index, filtered_meta)
        
        if results:
            st.subheader(f" Хамгийн төстэй {len(results)} үр дүн (Euclidean Distance):")
            
            for rank_data in results:
                st.markdown(f"###  №{rank_data['rank']} - {rank_data['title']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(
                        IMAGE_DIR / rank_data['name'], 
                        # Оноог "Зай (Distance)" гэж харуулна
                        caption=f"Зай (Distance): {rank_data['score']:.4f}", 
                        width=300
                    )
                
                with col2:
                    # Зайг харуулна
                    st.markdown(f"**Euclidean Distance (Зай):** `{rank_data['score']:.4f}` **(Бага байх тусам сайн)**") 
                    st.markdown(f"**Нэр:** `{rank_data['title']}`")
                    st.markdown(f"**Зохиогч:** `{rank_data['creator']}`")
                    st.info(rank_data['description'])
                
                st.markdown("---")

st.header("2. Текстээр Хайх (Нэр, Уран бүтээлч)")
# ... (Текстийн хайлт хэвээр үлдэнэ)
query_name = st.text_input("Зургийн нэр, уран бүтээлч эсвэл файлыг оруулна уу", key="text_query")

if query_name:
    # Текстийн хайлтыг хийх
    matched_rows = filtered_meta[
        filtered_meta['name'].str.contains(query_name, case=False) |
        filtered_meta['title'].str.contains(query_name, case=False) |
        filtered_meta['creator'].str.contains(query_name, case=False)
    ]
    
    if not matched_rows.empty:
        st.success(f"{len(matched_rows)} зураг олдлоо (Нэр болон уран бүтээлчээр):")         
        for rank, (_, row) in enumerate(matched_rows.iterrows(), start=1):
            st.markdown(f"### №{rank}: {row['title']}")
            text_col1, text_col2 = st.columns(2)
            art_path = IMAGE_DIR / row['name']
            with text_col1:
                if art_path.exists():
                    st.image(str(art_path), caption=f"{row['title']} ({row['creator']})", width=300)
                else:
                    st.error(f"Файл олдсонгүй: {row['name']}")
            
            with text_col2:
                st.markdown(f"**Нэр:** `{row['title']}`")
                st.markdown(f"**Зохиогч:** `{row['creator']}`")
                st.markdown("---")
                st.info(row['description'])
            st.markdown("---") 
    else:
        st.warning("Ийм нэртэй, уран бүтээлчтэй зураг олдсонгүй.")