# app.py (simplified)
import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json

# Google Drive IDs
MODEL_FILE_ID = "1bCyAfsonw3ef3Ig_KbKCHntCbL9T7yHN"
CLASS_FILE_ID = "19z1A68pyZLvHhaExmeheGI_Fq7qsF4jJ"

MODEL_PATH = "tomato_model.h5"
CLASS_PATH = "class_indices.json"

# Download model
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

# Download class indices
if not os.path.exists(CLASS_PATH):
    gdown.download(f"https://drive.google.com/uc?id={CLASS_FILE_ID}", CLASS_PATH, quiet=False)

@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model(MODEL_PATH)   # ✅ full model
    with open(CLASS_PATH) as f:
        class_indices = json.load(f)
    return model, class_indices

model, class_indices = load_trained_model()
class_names = {v: k for k, v in class_indices.items()}

# UI
st.title("🍅 Tomato Plant Disease Detection")
uploaded_file = st.file_uploader("Upload tomato leaf...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)
    st.success(f"Prediction: **{class_names[idx]}** ({confidence:.2f}%)")
