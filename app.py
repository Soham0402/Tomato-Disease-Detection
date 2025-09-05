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
# Google Drive File IDs (replace with yours)
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
# Disease Info (Description + Cure)
# =========================
disease_info = {
    "Late_blight": {
        "description": "Late blight is caused by the water mold *Phytophthora infestans*. It leads to large, dark, greasy lesions on leaves and fruits.",
        "cure": "Use fungicides like chlorothalonil or copper-based sprays. Remove infected plants and avoid overhead watering."
    },
    "healthy": {
        "description": "The plant leaf appears healthy without visible disease symptoms.",
        "cure": "No treatment needed. Maintain regular watering, fertilization, and pest monitoring."
    },
    "Early_blight": {
        "description": "Early blight is caused by *Alternaria solani*. It causes concentric ring spots on older leaves.",
        "cure": "Apply fungicides (mancozeb, chlorothalonil). Remove infected leaves and rotate crops."
    },
    "Septoria_leaf_spot": {
        "description": "Caused by *Septoria lycopersici*. Small, circular spots with dark borders appear on leaves.",
        "cure": "Remove infected leaves, improve air circulation, and apply fungicides."
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Transmitted by whiteflies, it causes upward curling of leaves and stunted growth.",
        "cure": "Control whiteflies with insecticides, remove infected plants, and use resistant varieties."
    },
    "Bacterial_spot": {
        "description": "Caused by *Xanthomonas* species, leading to dark, water-soaked lesions on leaves and fruits.",
        "cure": "Apply copper-based bactericides. Avoid overhead irrigation and rotate crops."
    },
    "Target_Spot": {
        "description": "Fungal disease caused by *Corynespora cassiicola*. It produces circular spots with concentric rings.",
        "cure": "Apply fungicides like mancozeb or chlorothalonil. Remove debris and infected leaves."
    },
    "Tomato_mosaic_virus": {
        "description": "Viral disease that causes mottling, mosaic patterns, and distorted leaves.",
        "cure": "No direct cure. Remove infected plants, disinfect tools, and use resistant varieties."
    },
    "Leaf_Mold": {
        "description": "Caused by *Passalora fulva*. It produces yellow spots on the upper leaf surface with fuzzy mold below.",
        "cure": "Increase ventilation, avoid overhead watering, and apply fungicides."
    },
    "Spider_mites Two-spotted_spider_mite": {
        "description": "Caused by spider mites, which suck sap from leaves, leading to yellow stippling and webbing.",
        "cure": "Spray miticides or neem oil. Wash plants with water to remove mites."
    },
    "Powdery Mildew": {
        "description": "Fungal disease characterized by white, powdery growth on leaves and stems.",
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
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((128, 128))  # same as training size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    disease_name = idx_to_class[predicted_class]

    st.subheader(f"🦠 Predicted Disease: **{disease_name}**")

    # Show description + cure
    if disease_name in disease_info:
        st.markdown(f"**Description:** {disease_info[disease_name]['description']}")
        st.markdown(f"**Recommended Cure:** {disease_info[disease_name]['cure']}")
    else:
        st.warning("No additional information available for this disease.")
