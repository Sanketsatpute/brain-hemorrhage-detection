import os
import cv2
import numpy as np
from tensorflow import keras

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "hemorrhage_model_v2.h5"
)

THRESHOLD = 0.41

IMAGE_PATHS = [
    (
        "049/1.jpg",
        os.path.join(
            BASE_DIR,
            "Patients_CT",
            "049",
            "brain",
            "1.jpg"
        )
    ),
    (
        "049/14.jpg",
        os.path.join(
            BASE_DIR,
            "Patients_CT",
            "049",
            "brain",
            "14.jpg"
        )
    )
]


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("=" * 70)
print("Loading Model V2")
print("=" * 70)

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_image(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # Resize
    image = cv2.resize(
        image,
        (224, 224)
    )

    # Convert grayscale → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB
    )

    # Convert to float
    image = image.astype(np.float32)

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        image,
        verbose=0
    )

    score = float(prediction[0][0])

    has_hemorrhage = score >= THRESHOLD

    return score, has_hemorrhage


# --------------------------------------------------
# Test images
# --------------------------------------------------

print("=" * 70)
print("Testing CT Images")
print("=" * 70)
print()

for name, path in IMAGE_PATHS:

    score, has_hemorrhage = predict_image(path)

    print(f"Image       : {name}")
    print(f"Score       : {score:.4f}")
    print(f"Threshold   : {THRESHOLD}")
    print(
        "Prediction  :",
        "Hemorrhage Detected"
        if has_hemorrhage
        else "No Hemorrhage Detected"
    )

    print("-" * 70)


print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)