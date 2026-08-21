# """
# Brain Hemorrhage Detection Model Training Script
# This script creates and trains a CNN model for detecting intracranial hemorrhage in CT scans.
# """

# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, models
# from sklearn.model_selection import train_test_split
# import os

# def create_model(input_shape=(224, 224, 1)):
#     """
#     Create a CNN model for brain hemorrhage detection
    
#     Args:
#         input_shape: Shape of input images (height, width, channels)
    
#     Returns:
#         Compiled Keras model
#     """
#     model = models.Sequential([
#         layers.Input(shape=input_shape),
        
#         # First Convolutional Block
#         layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),
#         layers.Dropout(0.25),
        
#         # Second Convolutional Block
#         layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),
#         layers.Dropout(0.25),
        
#         # Third Convolutional Block
#         layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),
#         layers.Dropout(0.25),
        
#         # Fourth Convolutional Block
#         layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),
#         layers.Dropout(0.25),
        
#         # Global Average Pooling
#         layers.GlobalAveragePooling2D(),
        
#         # Dense Layers
#         layers.Dense(512, activation='relu'),
#         layers.BatchNormalization(),
#         layers.Dropout(0.5),
        
#         layers.Dense(256, activation='relu'),
#         layers.BatchNormalization(),
#         layers.Dropout(0.5),
        
#         # Output Layer (Binary Classification)
#         layers.Dense(1, activation='sigmoid')
#     ])
    
#     # Compile the model
#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=1e-4),
#         loss='binary_crossentropy',
#         metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
#     )
    
#     return model


# def create_dummy_model():
#     """
#     Create and save a pre-trained dummy model for demonstration
#     This model can be used for testing without having actual training data
#     """
#     print("Creating pre-trained model for brain hemorrhage detection...")
    
#     model = create_model()
    
#     # Generate synthetic data for initial training (just to initialize weights properly)
#     X_dummy = np.random.randn(100, 224, 224, 1).astype(np.float32)
#     y_dummy = np.random.randint(0, 2, (100,))
    
#     # Quick training to initialize weights
#     model.train_on_batch(X_dummy[:32], y_dummy[:32])
    
#     # Save the model
#     model_path = os.path.join(os.path.dirname(__file__), 'hemorrhage_model.h5')
#     model.save(model_path)
#     print(f"Model saved to {model_path}")
    
#     # Print model summary
#     print("\nModel Architecture:")
#     model.summary()
    
#     return model_path


# if __name__ == "__main__":
#     create_dummy_model()

"""
Brain Hemorrhage Detection - Real CNN Training

Trains a binary classifier using the prepared CT dataset:

dataset/
├── train/
│   ├── normal/
│   └── hemorrhage/
├── validation/
│   ├── normal/
│   └── hemorrhage/
└── test/
    ├── normal/
    └── hemorrhage/
"""

import os
import json
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# MODEL_PATH = os.path.join(
#     MODEL_DIR,
#     "hemorrhage_model.keras"
# )

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hemorrhage_model.h5"
)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

SEED = 42

# ============================================================
# Reproducibility
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# Check dataset
# ============================================================

print("=" * 70)
print("Brain Hemorrhage Detection - Model Training")
print("=" * 70)

for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"Dataset directory not found:\n{directory}\n\n"
            "Run prepare_dataset.py first."
        )

# ============================================================
# Load datasets
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["normal", "hemorrhage"],
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["normal", "hemorrhage"],
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["normal", "hemorrhage"],
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass names:")
print(train_ds.class_names)

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

normal_count = len(
    os.listdir(
        os.path.join(TRAIN_DIR, "normal")
    )
)

hemorrhage_count = len(
    os.listdir(
        os.path.join(TRAIN_DIR, "hemorrhage")
    )
)

print("\nTraining class distribution:")
print(f"Normal      : {normal_count}")
print(f"Hemorrhage  : {hemorrhage_count}")

classes = np.array([0, 1])

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.array(
        [0] * normal_count +
        [1] * hemorrhage_count
    )
)

class_weights = {
    0: float(class_weights_array[0]),
    1: float(class_weights_array[1])
}

print("\nClass weights:")
print(f"Normal      : {class_weights[0]:.4f}")
print(f"Hemorrhage  : {class_weights[1]:.4f}")

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
    ],
    name="data_augmentation"
)

# ============================================================
# Create CNN
# ============================================================

def create_model():

    inputs = keras.Input(
        shape=(224, 224, 1),
        name="ct_scan"
    )

    # Normalize pixel values
    x = layers.Rescaling(
        1.0 / 255.0
    )(inputs)

    # Data augmentation
    x = data_augmentation(x)

    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(0.25)(x)

    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(0.25)(x)

    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(0.30)(x)

    # --------------------------------------------------------
    # Block 4
    # --------------------------------------------------------

    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(0.30)(x)

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="hemorrhage_probability"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    return model


model = create_model()

# ============================================================
# Compile
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        keras.metrics.Precision(
            name="precision"
        ),
        keras.metrics.Recall(
            name="recall"
        ),
        keras.metrics.AUC(
            name="auc"
        )
    ]
)

print("\nModel architecture:")
model.summary()

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
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================
# Train
# ============================================================

print("\n")
print("=" * 70)
print("Starting training...")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ============================================================
# Save training history
# ============================================================

history_path = os.path.join(
    MODEL_DIR,
    "training_history.json"
)

with open(history_path, "w") as f:

    json.dump(
        {
            key: [
                float(value)
                for value in values
            ]
            for key, values in history.history.items()
        },
        f,
        indent=4
    )

# ============================================================
# Load best model
# ============================================================

print("\nLoading best model...")

best_model = keras.models.load_model(
    MODEL_PATH
)

# ============================================================
# Evaluate test dataset
# ============================================================

print("\n")
print("=" * 70)
print("Evaluating on TEST dataset")
print("=" * 70)

test_results = best_model.evaluate(
    test_ds,
    verbose=1
)

for name, value in zip(
    best_model.metrics_names,
    test_results
):
    print(f"{name}: {value:.4f}")

# ============================================================
# Generate predictions
# ============================================================

print("\nGenerating test predictions...")

y_true = []
y_probability = []

for images, labels in test_ds:

    predictions = best_model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        labels.numpy().flatten().astype(int)
    )

    y_probability.extend(
        predictions.flatten()
    )

y_true = np.array(y_true)
y_probability = np.array(y_probability)

y_pred = (
    y_probability >= 0.5
).astype(int)

# ============================================================
# Classification report
# ============================================================

print("\n")
print("=" * 70)
print("Classification Report")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Normal",
            "Hemorrhage"
        ],
        digits=4
    )
)

# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# ============================================================
# ROC-AUC
# ============================================================

auc = roc_auc_score(
    y_true,
    y_probability
)

print(f"\nROC-AUC: {auc:.4f}")

# ============================================================
# Final
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"\nModel saved to:")
print(MODEL_PATH)

print(f"\nTraining history saved to:")
print(history_path)

print("\nYou can now connect this model to Flask.")