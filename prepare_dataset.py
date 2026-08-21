import os
import shutil
import random
import pandas as pd
import cv2

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "hemorrhage_diagnosis.csv")
PATIENTS_DIR = os.path.join(BASE_DIR, "Patients_CT")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset")

IMAGE_SIZE = (224, 224)

# Patient-level split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ============================================================
# Helper functions
# ============================================================

def create_directories():
    directories = [
        os.path.join(OUTPUT_DIR, "train", "normal"),
        os.path.join(OUTPUT_DIR, "train", "hemorrhage"),

        os.path.join(OUTPUT_DIR, "validation", "normal"),
        os.path.join(OUTPUT_DIR, "validation", "hemorrhage"),

        os.path.join(OUTPUT_DIR, "test", "normal"),
        os.path.join(OUTPUT_DIR, "test", "hemorrhage"),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_label(row):
    """
    Convert CSV diagnosis into binary classification.

    No_Hemorrhage = 1 -> Normal
    No_Hemorrhage = 0 -> Hemorrhage
    """

    if row["No_Hemorrhage"] == 1:
        return 0

    return 1


def get_patient_ids(df):
    return sorted(df["PatientNumber"].unique())


def split_patients(patient_ids):
    random.seed(RANDOM_SEED)

    patient_ids = list(patient_ids)
    random.shuffle(patient_ids)

    total = len(patient_ids)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_patients = patient_ids[:train_end]
    val_patients = patient_ids[train_end:val_end]
    test_patients = patient_ids[val_end:]

    return train_patients, val_patients, test_patients


def process_image(image_path, output_path):
    """
    Read CT image, convert to grayscale and resize to 224x224.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"WARNING: Could not read {image_path}")
        return False

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    cv2.imwrite(output_path, image)

    return True


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Brain Hemorrhage Dataset Preparation")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Could not find CSV file:\n{CSV_PATH}"
        )

    if not os.path.exists(PATIENTS_DIR):
        raise FileNotFoundError(
            f"Could not find Patients_CT directory:\n{PATIENTS_DIR}"
        )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    print("\nReading diagnosis CSV...")

    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "PatientNumber",
        "SliceNumber",
        "No_Hemorrhage"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' not found in CSV."
            )

    print(f"Total CSV records: {len(df)}")

    # --------------------------------------------------------
    # Patient IDs
    # --------------------------------------------------------

    patient_ids = get_patient_ids(df)

    print(f"Total patients: {len(patient_ids)}")

    # --------------------------------------------------------
    # Split patients
    # --------------------------------------------------------

    train_patients, val_patients, test_patients = split_patients(
        patient_ids
    )

    print("\nPatient split:")
    print(f"Training patients   : {len(train_patients)}")
    print(f"Validation patients : {len(val_patients)}")
    print(f"Test patients       : {len(test_patients)}")

    print("\nTraining patient IDs:")
    print(train_patients)

    print("\nValidation patient IDs:")
    print(val_patients)

    print("\nTest patient IDs:")
    print(test_patients)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # Process dataset
    # --------------------------------------------------------

    splits = {
        "train": train_patients,
        "validation": val_patients,
        "test": test_patients
    }

    statistics = {
        "train": {"normal": 0, "hemorrhage": 0},
        "validation": {"normal": 0, "hemorrhage": 0},
        "test": {"normal": 0, "hemorrhage": 0}
    }

    print("\nProcessing CT images...\n")

    for split_name, patients in splits.items():

        print(f"\nProcessing {split_name.upper()}...")

        split_df = df[df["PatientNumber"].isin(patients)]

        for _, row in split_df.iterrows():

            patient_number = int(row["PatientNumber"])
            slice_number = int(row["SliceNumber"])

            label = get_label(row)

            if label == 0:
                class_name = "normal"
            else:
                class_name = "hemorrhage"

            patient_folder = f"{patient_number:03d}"

            image_path = os.path.join(
                PATIENTS_DIR,
                patient_folder,
                "brain",
                f"{slice_number}.jpg"
            )

            if not os.path.exists(image_path):
                print(
                    f"WARNING: Missing image: {image_path}"
                )
                continue

            output_filename = (
                f"patient_{patient_number:03d}"
                f"_slice_{slice_number:03d}.png"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                split_name,
                class_name,
                output_filename
            )

            success = process_image(
                image_path,
                output_path
            )

            if success:
                statistics[split_name][class_name] += 1

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)

    for split_name in ["train", "validation", "test"]:

        normal_count = statistics[split_name]["normal"]
        hemorrhage_count = statistics[split_name]["hemorrhage"]

        total = normal_count + hemorrhage_count

        print(f"\n{split_name.upper()}")

        print(f"  Normal      : {normal_count}")
        print(f"  Hemorrhage  : {hemorrhage_count}")
        print(f"  Total       : {total}")

    print("\nDataset created at:")
    print(OUTPUT_DIR)

    print("\nStructure:")
    print("""
dataset/
├── train/
│   ├── normal/
│   └── hemorrhage/
│
├── validation/
│   ├── normal/
│   └── hemorrhage/
│
└── test/
    ├── normal/
    └── hemorrhage/
""")

    print("=" * 70)


if __name__ == "__main__":
    main()