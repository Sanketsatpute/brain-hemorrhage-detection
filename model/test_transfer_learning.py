import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

print("=" * 70)
print("TRANSFER LEARNING SANITY CHECK")
print("=" * 70)

print("TensorFlow:", tf.__version__)
print()

# Load a small portion of the dataset
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

print()
print("Classes:", train_ds.class_names)

# Get one batch
images, labels = next(iter(train_ds))

print()
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print()
print("Loading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation="relu")(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = keras.Model(inputs, outputs)

print()
print("Running one prediction batch...")

predictions = model(images, training=False)

print()
print("Prediction shape:", predictions.shape)

print(
    "Prediction range:",
    float(tf.reduce_min(predictions)),
    "to",
    float(tf.reduce_max(predictions))
)

print()
print("=" * 70)
print("SANITY CHECK PASSED")
print("=" * 70)