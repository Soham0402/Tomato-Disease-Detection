# app.py
import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json

# Google Drive File ID for full model (not just weights)
FILE_ID = "1bCyAfsonw3ef3Ig_KbKCHntCbL9T7yHN"
OUTPUT_PATH = "tomato_model.h5"

url = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model if not already present
if not os.path.exists(OUTPUT_PATH):
    gdown.download(url, OUTPUT_PATH, quiet=False)

# Download class indices json (upload it to Google Drive too and use its FILE_ID)
CLASS_FILE_ID = "PUT_YOUR_CLASS_INDICES_JSON_FILE_ID_HERE"
CLASS_OUTPUT = "class_indices.json"

if not os.path.exists(CLASS_OUTPUT):
    gdown.download(f"https://drive.google.com/uc?id={CLASS_FILE_ID}", CLASS_OUTPUT, quiet=False)

# ✅ Load model directly
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model(OUTPUT_PATH)
    with open(CLASS_OUTPUT) as f:
        class_indices = json.load(f)
    return model, class_indices

model, class_indices = load_trained_model()
class_names = {v: k for k, v in class_indices.items()}

# Streamlit UI
st.title("🍅 Tomato Plant Disease Detection")
st.write("Upload a tomato leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result_idx = np.argmax(prediction)
    result = class_names[result_idx]
    confidence = np.max(prediction) * 100

    st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)
    st.success(f"Prediction: **{result}** ({confidence:.2f}%)")
