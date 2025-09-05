
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
import ast
from PIL import Image

# =========================
# Google Drive File IDs
# =========================
MODEL_FILE_ID = "1tw3IHJfdxnCX-vPl7N3pED_uxzkgdNop"
CLASS_FILE_ID = "1uZv2WOdDPMgWSQAdoXmHiwyhfYPa8o0I"

MODEL_PATH = "tomato_cnn.h5"
CLASS_FILE = "class_indices.txt"

# =========================
# Download model + class map if not present
# =========================
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)

if not os.path.exists(CLASS_FILE):
    gdown.download(f"https://drive.google.com/uc?id={CLASS_ID}", CLASS_FILE, quiet=False)

# =========================
# Load model and class indices
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_FILE, "r") as f:
    class_indices = ast.literal_eval(f.read())

# Reverse mapping (index → class name)
idx_to_class = {v: k for k, v in class_indices.items()}

# =========================
# Disease Info
# =========================
disease_info = {
    "Late_blight": {
        "description": "Late blight is caused by *Phytophthora infestans*. Large, dark, greasy lesions on leaves and fruits.",
        "cure": "Use fungicides like chlorothalonil or copper-based sprays. Remove infected plants and avoid overhead watering."
    },
    "healthy": {
        "description": "Leaf appears healthy without visible disease.",
        "cure": "No treatment needed. Maintain good watering and fertilization."
    },
    "Early_blight": {
        "description": "Caused by *Alternaria solani*. Produces concentric ring spots on older leaves.",
        "cure": "Apply fungicides (mancozeb, chlorothalonil). Remove infected leaves and rotate crops."
    },
    "Septoria_leaf_spot": {
        "description": "Caused by *Septoria lycopersici*. Small, circular spots with dark borders.",
        "cure": "Remove infected leaves, improve air circulation, and apply fungicides."
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Viral disease spread by whiteflies. Leaves curl upwards, growth is stunted.",
        "cure": "Control whiteflies with insecticides. Remove infected plants and use resistant varieties."
    },
    "Bacterial_spot": {
        "description": "Caused by *Xanthomonas*. Leads to dark, water-soaked lesions.",
        "cure": "Use copper-based bactericides. Avoid overhead irrigation."
    },
    "Target_Spot": {
        "description": "Caused by *Corynespora cassiicola*. Produces circular spots with concentric rings.",
        "cure": "Apply fungicides like mancozeb or chlorothalonil. Remove infected debris."
    },
    "Tomato_mosaic_virus": {
        "description": "Viral disease that causes mottling, mosaic patterns, and distorted leaves.",
        "cure": "No cure. Remove infected plants, disinfect tools, and use resistant seeds."
    },
    "Leaf_Mold": {
        "description": "Caused by *Passalora fulva*. Yellow spots above, fuzzy mold below.",
        "cure": "Increase ventilation, avoid overhead watering, and apply fungicides."
    },
    "Spider_mites_Two-spotted_spider_mite": {
        "description": "Spider mites suck sap, leading to yellow stippling and webbing.",
        "cure": "Spray miticides or neem oil. Wash plants with water to remove mites."
    },
    "Powdery_Mildew": {
        "description": "White, powdery growth on leaves and stems.",
        "cure": "Apply sulfur-based fungicides, neem oil, or potassium bicarbonate sprays."
    }
}

# =========================
# Streamlit UI
# =========================
st.title("🌱 Tomato Plant Disease Detection")
st.write("Upload a tomato leaf image to detect disease and get cure recommendations.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    disease_name = idx_to_class[predicted_class]

    # 🔹 Normalize disease_name (replace spaces with underscores)
    disease_key = disease_name.replace(" ", "_")

    st.subheader(f"🦠 Predicted Disease: **{disease_name}**")

    # Print description + cure
    if disease_key in disease_info:
        st.markdown(f"**Description:** {disease_info[disease_key]['description']}")
        st.markdown(f"**Recommended Cure:** {disease_info[disease_key]['cure']}")
    else:
        st.error(f"No details found for: {disease_name}")
