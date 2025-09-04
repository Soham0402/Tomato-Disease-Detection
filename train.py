# train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Paths
base_path = r"C:/Users/Soham/Downloads/Tomato_dataset"
train_dir = os.path.join(base_path, "train")
valid_dir = os.path.join(base_path, "valid")

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 48
EPOCHS = 10

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_gen = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Build Model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(train_gen, validation_data=valid_gen, epochs=EPOCHS)

# Save
os.makedirs("models", exist_ok=True)
model.save("models/tomato_model.h5")
print("✅ Model saved at models/tomato_model.h5")

# Save class indices (mapping class names)
import json
with open("models/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)
print("✅ Class indices saved")
