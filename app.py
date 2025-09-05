import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import ast
import os

# Google Drive File IDs (replace with yours)
MODEL_FILE_ID = "1tw3IHJfdxnCX-vPl7N3pED_uxzkgdNop"
CLASS_FILE_ID = "1uZv2WOdDPMgWSQAdoXmHiwyhfYPa8o0I"

MODEL_PATH = "tomato_cnn.h5"
CLASS_FILE = "class_indices.txt"

# Download from Google Drive if not exists
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

if not os.path.exists(CLASS_FILE):
    gdown.download(f"https://drive.google.com/uc?id={CLASS_FILE_ID}", CLASS_FILE, quiet=False)

# Load Model + Class Indices
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_FILE, "r") as f:
        class_indices = ast.literal_eval(f.read())
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_model_and_classes()

# Streamlit UI
st.title("🍅 Tomato Disease Detection (Custom CNN)")
st.write("Upload a tomato leaf image to detect disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    st.success(f"Prediction: **{idx_to_class[pred_class]}** ({confidence:.2f} confidence)")
