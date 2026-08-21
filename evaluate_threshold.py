import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "model/hemorrhage_model.h5"

VAL_DIR = "dataset/validation"
TEST_DIR = "dataset/test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# Load model
# ============================================================

print("=" * 70)
print("Loading trained model...")
print("=" * 70)

model = load_model(MODEL_PATH)

print("Model loaded successfully.")
print()


# ============================================================
# Load validation dataset
# IMPORTANT:
# 0 = normal
# 1 = hemorrhage
# ============================================================

print("Loading validation dataset...")

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

print("Classes:", val_ds.class_names)
print()


# ============================================================
# Load test dataset
# ============================================================

print("Loading test dataset...")

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

print("Classes:", test_ds.class_names)
print()


# ============================================================
# Get validation predictions
# ============================================================

print("=" * 70)
print("Generating validation predictions...")
print("=" * 70)

val_predictions = model.predict(val_ds)

# Convert shape:
# [[0.72], [0.31], [0.91]]
# into:
# [0.72, 0.31, 0.91]

val_predictions = val_predictions.ravel()

# Get actual labels
val_labels = np.concatenate([
    y.numpy().ravel()
    for x, y in val_ds
]).astype(int)

print("Validation images:", len(val_labels))

print(
    "Prediction range:",
    round(float(val_predictions.min()), 4),
    "to",
    round(float(val_predictions.max()), 4)
)

print()

normal_predictions = val_predictions[val_labels == 0]
hemorrhage_predictions = val_predictions[val_labels == 1]

print("Normal prediction statistics:")
print(
    "Min:",
    np.min(normal_predictions),
    "Max:",
    np.max(normal_predictions),
    "Mean:",
    np.mean(normal_predictions)
)

print()

print("Hemorrhage prediction statistics:")
print(
    "Min:",
    np.min(hemorrhage_predictions),
    "Max:",
    np.max(hemorrhage_predictions),
    "Mean:",
    np.mean(hemorrhage_predictions)
)

print()
# ============================================================
# Validation ROC-AUC
# ============================================================

val_auc = roc_auc_score(
    val_labels,
    val_predictions
)

print("Validation ROC-AUC:", round(val_auc, 4))
print()


# ============================================================
# Test different thresholds
# ============================================================

print("=" * 70)
print("Testing different classification thresholds")
print("=" * 70)

print()

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 60)

best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.10, 0.96, 0.05):

    # Search thresholds specifically across the range
    # produced by the model.

    thresholds = np.linspace(
        float(val_predictions.min()),
        float(val_predictions.max()),
        1000
    )

    best_threshold = 0.5
    best_f1 = 0

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 60)

for threshold in thresholds:

    predictions = (
        val_predictions >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        val_labels,
        predictions
    )

    precision = precision_score(
        val_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        val_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        val_labels,
        predictions,
        zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


# ============================================================
# Best threshold
# ============================================================

print()
print("=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"Best threshold: {best_threshold:.2f}"
)

print(
    f"Best validation F1: {best_f1:.4f}"
)

print()


# ============================================================
# Evaluate best threshold on validation set
# ============================================================

val_best_predictions = (
    val_predictions >= best_threshold
).astype(int)

print("=" * 70)
print("Validation results using best threshold")
print("=" * 70)

print(
    classification_report(
        val_labels,
        val_best_predictions,
        target_names=["Normal", "Hemorrhage"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        val_labels,
        val_best_predictions
    )
)

print()


# ============================================================
# NOW evaluate on completely unseen TEST data
# ============================================================

print("=" * 70)
print("Generating TEST predictions...")
print("=" * 70)

test_predictions = model.predict(test_ds)

test_predictions = test_predictions.ravel()

test_labels = np.concatenate([
    y.numpy().ravel()
    for x, y in test_ds
]).astype(int)


# IMPORTANT:
# Use the threshold selected from VALIDATION.
# Do NOT select a threshold using the test set.

test_class_predictions = (
    test_predictions >= best_threshold
).astype(int)


# ============================================================
# Test metrics
# ============================================================

test_accuracy = accuracy_score(
    test_labels,
    test_class_predictions
)

test_precision = precision_score(
    test_labels,
    test_class_predictions,
    zero_division=0
)

test_recall = recall_score(
    test_labels,
    test_class_predictions,
    zero_division=0
)

test_f1 = f1_score(
    test_labels,
    test_class_predictions,
    zero_division=0
)

test_auc = roc_auc_score(
    test_labels,
    test_predictions
)


print()
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {test_accuracy:.4f}"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1 Score  : {test_f1:.4f}"
)

print(
    f"ROC-AUC   : {test_auc:.4f}"
)

print()


# ============================================================
# Classification report
# ============================================================

print("=" * 70)
print("TEST CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        test_labels,
        test_class_predictions,
        target_names=["Normal", "Hemorrhage"],
        zero_division=0
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

print("Confusion Matrix:")
print(
    confusion_matrix(
        test_labels,
        test_class_predictions
    )
)

print()
print("=" * 70)
print("THRESHOLD EVALUATION COMPLETE")
print("=" * 70)