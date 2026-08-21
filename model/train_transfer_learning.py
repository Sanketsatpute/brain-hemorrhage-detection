"""
Brain Hemorrhage Detection - Transfer Learning Model

Model 2:
MobileNetV2 pretrained on ImageNet, fine-tuned for
normal vs intracranial hemorrhage classification.
"""

import os
import json
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "hemorrhage_model_v2.h5"
)

print("=" * 70)
print("Brain Hemorrhage Detection - Transfer Learning")
print("=" * 70)

print()
print("TensorFlow version:", tf.__version__)
print("Training directory:", TRAIN_DIR)
print("Validation directory:", VAL_DIR)
print("Test directory:", TEST_DIR)
print()


# ============================================================
# Load datasets
# ============================================================

print("=" * 70)
print("Loading datasets...")
print("=" * 70)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["normal", "hemorrhage"],
    color_mode="rgb",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["normal", "hemorrhage"],
    color_mode="rgb",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["normal", "hemorrhage"],
    color_mode="rgb",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print()
print("Classes:", train_ds.class_names)
print()


# ============================================================
# Performance optimization
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# Calculate class weights
# ============================================================

print("=" * 70)
print("Calculating class weights...")
print("=" * 70)

normal_count = 1553
hemorrhage_count = 180

total = normal_count + hemorrhage_count

class_weights = {
    0: total / (2 * normal_count),
    1: total / (2 * hemorrhage_count)
}

print("Class weights:")
print("Normal     :", class_weights[0])
print("Hemorrhage :", class_weights[1])
print()


# ============================================================
# Data augmentation
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(
            height_factor=0.03,
            width_factor=0.03
        ),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)


# ============================================================
# Build MobileNetV2 model
# ============================================================

print("=" * 70)
print("Building MobileNetV2 model...")
print("=" * 70)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False


inputs = keras.Input(
    shape=(224, 224, 3),
    name="ct_scan"
)

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="hemorrhage_probability"
)(x)

model = keras.Model(
    inputs,
    outputs
)


# ============================================================
# Compile
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc")
    ]
)

print()
model.summary()
print()


# ============================================================
# Callbacks
# ============================================================

callbacks = [

    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# Stage 1 — Train classification head
# ============================================================

print("=" * 70)
print("STAGE 1")
print("Training classification head...")
print("=" * 70)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# ============================================================
# Stage 2 — Fine-tune MobileNetV2
# ============================================================

print()
print("=" * 70)
print("STAGE 2")
print("Fine-tuning MobileNetV2...")
print("=" * 70)

base_model.trainable = True

# Freeze the first ~100 layers.
# Only the later feature extraction layers will be fine-tuned.
for layer in base_model.layers[:100]:
    layer.trainable = False


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc")
    ]
)


history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# ============================================================
# Load best model
# ============================================================

print()
print("=" * 70)
print("Loading best model...")
print("=" * 70)

model = keras.models.load_model(MODEL_PATH)


# ============================================================
# Evaluate test set
# ============================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

results = model.evaluate(
    test_ds,
    verbose=1
)

for name, value in zip(
    model.metrics_names,
    results
):
    print(f"{name}: {value:.4f}")


# ============================================================
# Save training information
# ============================================================

history = {
    "stage1": {
        key: [float(x) for x in values]
        for key, values in history1.history.items()
    },

    "stage2": {
        key: [float(x) for x in values]
        for key, values in history2.history.items()
    }
}

history_path = os.path.join(
    BASE_DIR,
    "model",
    "training_history_v2.json"
)

with open(
    history_path,
    "w"
) as f:
    json.dump(
        history,
        f,
        indent=2
    )


# ============================================================
# Complete
# ============================================================

print()
print("=" * 70)
print("TRANSFER LEARNING TRAINING COMPLETE")
print("=" * 70)

print()
print("Model saved to:")
print(MODEL_PATH)

print()
print("Training history saved to:")
print(history_path)

print()
print("This is Model V2.")
print("The original model has NOT been modified.")