import os
import shutil
import random

SOURCE_DIR = "data/rvf10k"
OUTPUT_DIR = "data/rvf10k_subset"

random.seed(42)

# Number of images per class
TRAIN_IMAGES = 1000
VALIDATION_IMAGES = 250
TEST_IMAGES = 250


def copy_images(source_folder, destination_folder, count):
    os.makedirs(destination_folder, exist_ok=True)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    selected = images[:count]

    for image in selected:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(destination_folder, image)
        )

    print(f"Copied {len(selected)} images from {source_folder}")


# Create train, validation and test folders
for split in ["train", "validation", "test"]:
    for label in ["real", "fake"]:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, label),
            exist_ok=True
        )


# TRAIN
for label in ["real", "fake"]:
    copy_images(
        os.path.join(SOURCE_DIR, "train", label),
        os.path.join(OUTPUT_DIR, "train", label),
        TRAIN_IMAGES
    )


# VALIDATION + TEST
# We take images from the original "valid" folder.
for label in ["real", "fake"]:

    source = os.path.join(SOURCE_DIR, "valid", label)

    images = [
        f for f in os.listdir(source)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    validation_images = images[:VALIDATION_IMAGES]
    test_images = images[
        VALIDATION_IMAGES:VALIDATION_IMAGES + TEST_IMAGES
    ]

    validation_path = os.path.join(
        OUTPUT_DIR, "validation", label
    )

    test_path = os.path.join(
        OUTPUT_DIR, "test", label
    )

    for image in validation_images:
        shutil.copy2(
            os.path.join(source, image),
            os.path.join(validation_path, image)
        )

    for image in test_images:
        shutil.copy2(
            os.path.join(source, image),
            os.path.join(test_path, image)
        )

    print(
        f"{label}: "
        f"{len(validation_images)} validation, "
        f"{len(test_images)} test"
    )


print("\nDataset subset created successfully!")
print(f"Location: {OUTPUT_DIR}")