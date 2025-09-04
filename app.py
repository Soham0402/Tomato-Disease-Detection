# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import json

# Load Model & Classes
model = load_model("models/tomato_model.h5")
with open("models/class_indices.json") as f:
    class_indices = json.load(f)

# Reverse dict: {0:"class_name"}
class_names = {v: k for k, v in class_indices.items()}

st.title("🍅 Tomato Plant Disease Detection")
st.write("Upload a tomato leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load & preprocess
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    result_idx = np.argmax(prediction)
    result = class_names[result_idx]
    confidence = np.max(prediction) * 100

    # Show result
    st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)
    st.success(f"Prediction: **{result}** ({confidence:.2f}%)")
