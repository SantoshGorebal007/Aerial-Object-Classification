import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os
from pathlib import Path
import traceback

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Aerial Bird vs Drone Classifier",
    page_icon="🛸",
    layout="centered"
)

IMG_SIZE = (224, 224)
CLASS_NAMES = ["bird", "drone"]

# Resolve project root no matter where Streamlit runs
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
models_dir = project_root / "models"

# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def list_models():
    """Return sorted list of available .h5 model files."""
    if not models_dir.exists():
        return []
    return sorted([f.name for f in models_dir.glob("*.h5")])

@st.cache_resource
def load_model_cached(model_path):
    """Load TF model with caching and safe error handling."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Unable to load model:\n```\n{traceback.format_exc()}\n```")
        return None

def preprocess_image(img):
    """Preprocess uploaded PIL image for TF model."""
    img = img.convert("RGB")
    img = ImageOps.fit(img, IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_label(model, img_tensor):
    """Run prediction and return label + prob."""
    preds = model.predict(img_tensor)
    prob = float(preds.ravel()[0])

    if prob >= 0.5:
        return "drone", prob
    else:
        return "bird", 1 - prob

# -------------------------------------------------------------
# UI HEADER
# -------------------------------------------------------------
st.title("🛸 Aerial Bird vs Drone Classifier")
st.write("Upload an aerial image and detect whether it contains a **Bird** or a **Drone**.")

st.markdown("---")

# -------------------------------------------------------------
# MODEL SELECTOR
# -------------------------------------------------------------
st.sidebar.header("📦 Model Selection")

available_models = list_models()

if available_models:
    selected_model_name = st.sidebar.selectbox(
        "Choose model from ../models",
        available_models
    )
    selected_model_path = models_dir / selected_model_name
else:
    st.sidebar.error("❌ No .h5 models found in ../models/")
    selected_model_path = None

if st.sidebar.button("Reload Model Cache"):
    st.cache_resource.clear()
    st.success("Model cache cleared!")

# -------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------
model = None
if selected_model_path is not None:
    st.sidebar.info(f"Selected model:\n`{selected_model_name}`")
    with st.spinner("Loading model…"):
        model = load_model_cached(str(selected_model_path))
        if model:
            st.sidebar.success("Model loaded successfully!")

st.markdown("---")

# -------------------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.subheader("📸 Uploaded Image")
    st.image(image, use_container_width=True)

    if model is None:
        st.error("⚠ No model loaded. Please select a model in the sidebar.")
    else:
        with st.spinner("Running prediction…"):
            try:
                img_tensor = preprocess_image(image)
                label, confidence = predict_label(model, img_tensor)

                st.markdown("## 🔍 Prediction Result")
                st.success(f"**Prediction:** `{label.upper()}`")
                st.write(f"**Confidence:** {confidence*100:.2f}%")

                st.progress(int(confidence * 100))

            except Exception:
                st.error("❌ Prediction failed.")
                st.text(traceback.format_exc())

else:
    st.info("👆 Upload an image to begin classification.")

st.markdown("---")
st.caption("Model auto-detected from `/models` folder.")
