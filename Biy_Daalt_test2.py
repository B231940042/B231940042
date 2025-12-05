import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Optional, List
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

# --- Config ---
IMAGE_DIR = Path("C:/Users/Lenovo/Documents/python/F.ITM324/src/famous_art_images")
TOP_K_DEFAULT = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
#  1. Load VGG16 Feature Extractor
# -----------------------------
@st.cache_resource
def load_vgg16_extractor():
    base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    base_model.eval()

    feature_extractor = torch.nn.Sequential(
        base_model.features,
        base_model.avgpool,
        torch.nn.Flatten()
    ).to(DEVICE)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return feature_extractor, preprocess


# -----------------------------
#  2. Extract feature from image
# -----------------------------
def extract_feature(image_path: Path, feature_extractor, preprocess) -> Optional[np.ndarray]:
    if not image_path.exists():
        return None

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = feature_extractor(x)

    return feat.cpu().numpy().flatten()


# -----------------------------
#  3. Build Feature Index
# -----------------------------
def build_index(metadata_df: pd.DataFrame, feature_extractor, preprocess) -> Dict[str, np.ndarray]:
    feature_index = {}

    for _, row in metadata_df.iterrows():
        img_path = IMAGE_DIR / row["name"]
        feat = extract_feature(img_path, feature_extractor, preprocess)
        if feat is not None:
            feature_index[row['id']] = feat

    return feature_index


# -----------------------------
#  4. Search Engine (Cosine Similarity)
# -----------------------------
def search_famous_art(query_feat: np.ndarray,
                      feature_index: Dict[str, np.ndarray],
                      metadata_df: pd.DataFrame,
                      top_k: int = TOP_K_DEFAULT):

    if len(feature_index) == 0:
        return "Индекс хоосон байна.", None

    indexed_ids = list(feature_index.keys())
    indexed_vectors = np.vstack([feature_index[_id] for _id in indexed_ids])

    sims = cosine_similarity(query_feat.reshape(1, -1), indexed_vectors).flatten()
    ranked_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for rank, ridx in enumerate(ranked_idx, start=1):
        doc_id = indexed_ids[ridx]
        score = float(sims[ridx])
        info = metadata_df[metadata_df['id'] == doc_id].iloc[0]

        results.append({
            "rank": rank,
            "score": score,
            "title": info['title'],
            "creator": info['creator'],
            "description": info['description'],
            "name": info['name']
        })

    return "Хайлт амжилттай.", results


# -----------------------------
#  5. Streamlit App UI
# -----------------------------
def main():
    st.title(" Зургийн төстэй байдлаар хайх систем (VGG16 + cosine similarity)")

    # Load model
    feature_extractor, preprocess = load_vgg16_extractor()

    # Load the REAL CSV FILE (art.csv)
    metadata_path = "C:\\Users\\Lenovo\\Documents\\python\\F.ITM324\\src\\art.csv"
    metadata_df = pd.read_csv(metadata_path)

    # Build index
    if st.button(" Индекс үүсгэх"):
        with st.spinner("Индекс үүсгэж байна..."):
            feature_index = build_index(metadata_df, feature_extractor, preprocess)
            st.session_state["feature_index"] = feature_index
        st.success("Индекс амжилттай үүслээ!")

    # Upload image
    uploaded_img = st.file_uploader("Зураг оруулна уу", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Оруулсан зураг", width=300)

        # Extract query feature
        with st.spinner("Feature гаргаж байна..."):
            x = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                query_feat = feature_extractor(x).cpu().numpy().flatten()

        # Check index
        if "feature_index" not in st.session_state:
            st.error("Эхлээд индекс үүсгэнэ үү!")
            return

        # Search
        status, results = search_famous_art(
            query_feat,
            st.session_state["feature_index"],
            metadata_df,
            top_k=3
        )

        if status != "Хайлт амжилттай.":
            st.error(status)
            return

        st.success("Top 3 төстэй зураг")

        for item in results:
            st.subheader(f"{item['rank']}. {item['title']} — {item['creator']}")
            st.write(f"Ижил төстэй оноо: **{item['score']:.4f}**")
            st.write(item['description'])

            img_path = IMAGE_DIR / item["name"]
            if img_path.exists():
                sim_img = Image.open(img_path)
                st.image(sim_img, caption=item['title'], width=300)
            else:
                st.warning(f"Зураг олдсонгүй: {img_path}")


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    main()
