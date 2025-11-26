"""
evaluate_yolo_knife.py

Runs evaluation for a YOLOv8 knife detector and prints:
- mAP50, mAP50-95, precision, recall, etc. (YOLO val metrics) using model.val()
- Image-level detection stats on the test set (which contains ONLY knife images):
    * total images
    * how many images had at least one predicted knife
    * how many were missed entirely
    * detection rate = detected / total
- Confusion matrix for (GT: knife present) vs (prediction: knife / no knife)
- Bar chart: detected vs missed images

REQUIREMENTS (install in your venv):
    pip install ultralytics matplotlib scikit-learn
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from ultralytics import YOLO


# ===================== CONFIG – EDIT THESE IF NEEDED =====================

# 1) Trained weights (desk-14)
WEIGHTS_PATH = r"C:\Users\desk-14\Desktop\ComputerVision\runs\train_20251126_205247\weights\best.pt"

# 2) data.yaml for your knife dataset (same one used for training)
DATA_YAML = r"C:\Users\desk-14\Desktop\ComputerVision\Dataset\data.yaml"

# 3) Dataset root + which split to evaluate on ("val" or "test")
DATASET_ROOT = r"C:\Users\desk-14\Desktop\ComputerVision\Dataset"
SPLIT = "test"  # "val" or "test" – here test = all knife images

IMG_SIZE = 512
CONF_THRESH = 0.25
IOU_THRESH = 0.5

# ========================================================================


def run_yolo_val(model: YOLO):
    """
    Use Ultralytics built-in validation to get:
    - mAP50, mAP50-95
    - Precision, Recall, etc.

    Also saves plots (PR curve, confusion matrix, etc.)
    to runs/detect/val/...
    """
    print("\n========== YOLO DETECTION METRICS (model.val) ==========")
    metrics = model.val(
        data=DATA_YAML,
        split=SPLIT,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        plots=True,   # saves PR curves, confusion matrix, etc.
        verbose=True,
    )

    print("\nRaw metrics dict:")
    try:
        results_dict = metrics.results_dict
    except AttributeError:
        results_dict = metrics.box.results_dict

    for k, v in results_dict.items():
        print(f"{k:20s}: {v:.4f}")


def image_level_detection_and_confusion(model: YOLO):
    """
    Test set contains ONLY knife images.

    For each image:
        - If there is at least one predicted box -> "detected" (prediction = 1)
        - If there are no predicted boxes        -> "missed"   (prediction = 0)

    Ground truth:
        - Every image has a knife -> GT label = 1 for all images.

    We report:
        - total_images
        - detected_images  (TP)
        - missed_images    (FN)
        - detection_rate = detected_images / total_images
        - confusion matrix (rows = GT, cols = pred) for class "knife present"
    """

    print("\n========== IMAGE-LEVEL DETECTION ON KNIFE-ONLY TEST SET ==========")

    images_dir = os.path.join(DATASET_ROOT, SPLIT, "images")

    # Run predictions on the whole images folder
    results = model.predict(
        source=images_dir,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save=False,
        verbose=False,
    )

    # Since every test image contains a knife, GT = 1 for all
    y_true = []
    y_pred = []

    detected = 0
    missed = 0

    for r in results:
        has_pred = r.boxes is not None and len(r.boxes) > 0

        # Ground truth: all images are knife images
        y_true.append(1)

        # Prediction: 1 if at least one box, else 0
        if has_pred:
            y_pred.append(1)
            detected += 1
        else:
            y_pred.append(0)
            missed += 1

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    total = len(y_true)
    if total == 0:
        print("No images found in:", images_dir)
        return

    detection_rate = detected / total

    print(f"Total knife images in {SPLIT}: {total}")
    print(f"Images with at least one predicted knife (TP): {detected}")
    print(f"Images with NO predicted knife (FN): {missed}")
    print(f"Image-level detection rate: {detection_rate:.4f}")

    # ---- Confusion matrix ----
    # Labels: 0 = 'GT no knife' (won't appear), 1 = 'GT knife'
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\nConfusion matrix (rows = GT, cols = prediction):")
    print(cm)
    print("Row 0 (GT = no knife) will be all zeros because test set is all knives.")
    print("Row 1 (GT = knife): [FN, TP] = [missed, detected].")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["GT: no knife", "GT: knife"],
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Knife-only Test Set)")
    plt.tight_layout()
    plt.show()

    # ---- Bar chart: detected vs missed ----
    labels = ["Detected (TP)", "Missed (FN)"]
    counts = [detected, missed]

    plt.figure()
    plt.bar(labels, counts)
    plt.title("Image-level Detection on Knife-only Test Set")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.show()


def main():
    print("Loading model...")
    model = YOLO(WEIGHTS_PATH)

    run_yolo_val(model)
    image_level_detection_and_confusion(model)


if __name__ == "__main__":
    main()
