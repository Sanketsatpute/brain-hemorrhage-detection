import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "hemorrhage_model_v2.h5"
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "validation"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# Load model
# ============================================================

print("=" * 70)
print("Loading Model V2...")
print("=" * 70)

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print()


# ============================================================
# Load datasets
# ============================================================

print("=" * 70)
print("Loading validation dataset...")
print("=" * 70)

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

print()

print("=" * 70)
print("Loading test dataset...")
print("=" * 70)

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


# ============================================================
# Generate validation predictions
# ============================================================

print("=" * 70)
print("Generating VALIDATION predictions...")
print("=" * 70)

val_predictions = model.predict(
    val_ds,
    verbose=1
).flatten()

val_labels = np.concatenate([
    y.numpy().flatten()
    for _, y in val_ds
])

print()
print("Validation images:", len(val_labels))

print(
    "Prediction range:",
    round(float(val_predictions.min()), 4),
    "to",
    round(float(val_predictions.max()), 4)
)

print()

# Separate predictions by class
normal_predictions = val_predictions[val_labels == 0]
hemorrhage_predictions = val_predictions[val_labels == 1]

print("Normal prediction statistics:")
print(
    "Min:",
    round(float(np.min(normal_predictions)), 4),
    "Max:",
    round(float(np.max(normal_predictions)), 4),
    "Mean:",
    round(float(np.mean(normal_predictions)), 4)
)

print()

print("Hemorrhage prediction statistics:")
print(
    "Min:",
    round(float(np.min(hemorrhage_predictions)), 4),
    "Max:",
    round(float(np.max(hemorrhage_predictions)), 4),
    "Mean:",
    round(float(np.mean(hemorrhage_predictions)), 4)
)

print()

# Validation AUC
val_auc = roc_auc_score(
    val_labels,
    val_predictions
)

print(
    "Validation ROC-AUC:",
    round(val_auc, 4)
)

print()


# ============================================================
# Find best threshold
# ============================================================

print("=" * 70)
print("Testing classification thresholds")
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


thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:

    val_predicted_classes = (
        val_predictions >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        val_labels,
        val_predicted_classes
    )

    precision = precision_score(
        val_labels,
        val_predicted_classes,
        zero_division=0
    )

    recall = recall_score(
        val_labels,
        val_predicted_classes,
        zero_division=0
    )

    f1 = f1_score(
        val_labels,
        val_predicted_classes,
        zero_division=0
    )

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
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
    "Best threshold:",
    round(float(best_threshold), 2)
)

print(
    "Best validation F1:",
    round(float(best_f1), 4)
)

print()


# ============================================================
# Validation evaluation
# ============================================================

val_final_predictions = (
    val_predictions >= best_threshold
).astype(int)

print("=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(
    classification_report(
        val_labels,
        val_final_predictions,
        target_names=[
            "Normal",
            "Hemorrhage"
        ],
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        val_labels,
        val_final_predictions
    )
)

print()


# ============================================================
# Generate TEST predictions
# ============================================================

print("=" * 70)
print("Generating TEST predictions...")
print("=" * 70)

test_predictions = model.predict(
    test_ds,
    verbose=1
).flatten()

test_labels = np.concatenate([
    y.numpy().flatten()
    for _, y in test_ds
])


# ============================================================
# Final test evaluation
# ============================================================

test_predicted_classes = (
    test_predictions >= best_threshold
).astype(int)

test_accuracy = accuracy_score(
    test_labels,
    test_predicted_classes
)

test_precision = precision_score(
    test_labels,
    test_predicted_classes,
    zero_division=0
)

test_recall = recall_score(
    test_labels,
    test_predicted_classes,
    zero_division=0
)

test_f1 = f1_score(
    test_labels,
    test_predicted_classes,
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
    "Threshold :",
    round(float(best_threshold), 2)
)

print(
    "Accuracy  :",
    round(test_accuracy, 4)
)

print(
    "Precision :",
    round(test_precision, 4)
)

print(
    "Recall    :",
    round(test_recall, 4)
)

print(
    "F1 Score  :",
    round(test_f1, 4)
)

print(
    "ROC-AUC   :",
    round(test_auc, 4)
)

print()


# ============================================================
# Test classification report
# ============================================================

print("=" * 70)
print("TEST CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        test_labels,
        test_predicted_classes,
        target_names=[
            "Normal",
            "Hemorrhage"
        ],
        zero_division=0
    )
)

print()

print("Confusion Matrix:")

print(
    confusion_matrix(
        test_labels,
        test_predicted_classes
    )
)

print()

print("=" * 70)
print("V2 THRESHOLD EVALUATION COMPLETE")
print("=" * 70)