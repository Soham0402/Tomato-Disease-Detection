import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Paths
base_path = r"C:/Users/Soham/Downloads/Tomato_dataset"
train_dir = os.path.join(base_path, "train")
valid_dir = os.path.join(base_path, "valid")

MODEL_PATH = "tomato_cnn.h5"
CLASS_FILE = "class_indices.txt"

# Image settings
IMG_SIZE = (128, 128)   # small size for faster training
BATCH_SIZE = 32

# Data generators with augmentation for train, only rescale for validation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Improved Small CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,   # increase a bit for stability
    verbose=1
)

# Save model
model.save(MODEL_PATH)

# Save class indices properly
with open(CLASS_FILE, "w") as f:
    for cls, idx in train_data.class_indices.items():
        f.write(f"{idx}:{cls}\n")

print("✅ Model and class indices saved successfully.")
print("👉 Upload `tomato_cnn.h5` and `class_indices.txt` to Google Drive for app.py")
