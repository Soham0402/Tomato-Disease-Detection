
# app.py
import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json

# Google Drive File ID (weights only)
FILE_ID = "1bCyAfsonw3ef3Ig_KbKCHntCbL9T7yHN"
OUTPUT_PATH = "tomato_weights.h5"

url = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model weights if not already present
if not os.path.exists(OUTPUT_PATH):
    gdown.download(url, OUTPUT_PATH, quiet=False)

# ✅ Rebuild the exact architecture you used during training
def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,   # we will load our own weights
        pooling="avg"
    )
    x = tf.keras.layers.Dense(128, activation="relu")(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model

@st.cache_resource
def load_trained_model():
    # Load class indices first to know number of classes
    with open("models/class_indices.json") as f:
        class_indices = json.load(f)

    num_classes = len(class_indices)
    model = build_model(num_classes)
    model.load_weights(OUTPUT_PATH)   # load weights into architecture
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
