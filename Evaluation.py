"""
evaluate_yolo_cigarette.py

Runs evaluation for a YOLOv8 cigarette detector and prints:
- mAP50, mAP50-95, precision, recall, etc. (YOLO val metrics) using model.val()
- Image-level metrics on the chosen split (val/test):
    * Accuracy, precision, recall, F1 for "cigarette present vs not present"
    * Confusion matrix (cigarette vs no-cigarette)
    * Bar chart: images with cigarette detected vs not detected

Ground truth per image:
    GT = 1 if label file exists and has at least one object (cigarette)
    GT = 0 if no label file or empty file

REQUIREMENTS (in your venv):
    pip install ultralytics matplotlib scikit-learn numpy
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ultralytics import YOLO


# ===================== CONFIG – EDIT THESE IF NEEDED =====================

# 1) Trained weights on the OTHER DEVICE (jadif)
WEIGHTS_PATH = r"C:\Users\jadif\OneDrive\Documentos\Desktop\CV\runs\train_20251126_231444\weights\best.pt"

# 2) data.yaml for your cigarette dataset (same one used for training)
DATA_YAML = r"C:\Users\jadif\OneDrive\Documentos\Desktop\CV\Dataset\data.yaml"

# 3) Dataset root + which split to evaluate on ("val" or "test")
DATASET_ROOT = r"C:\Users\jadif\OneDrive\Documentos\Desktop\CV\Dataset"
SPLIT = "test"  # "val" or "test"

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


def image_level_metrics_and_confusion(model: YOLO):
    """
    Image-level evaluation for cigarette vs no-cigarette.

    Ground truth per image:
        GT = 1 (cigarette present) if the label file exists and has at least one line
        GT = 0 (no-cigarette)      if the label file is missing or empty

    Prediction per image:
        pred = 1 if model predicts at least one box
        pred = 0 if model predicts no boxes
    """

    print("\n========== IMAGE-LEVEL METRICS (cigarette vs no-cigarette) ==========")

    images_dir = os.path.join(DATASET_ROOT, SPLIT, "images")
    labels_dir = os.path.join(DATASET_ROOT, SPLIT, "labels")

    # Run predictions on the whole images folder
    results = model.predict(
        source=images_dir,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save=False,
        verbose=False,
    )

    y_true = []  # 1 = GT has cigarette, 0 = GT no cigarette
    y_pred = []  # 1 = predicted cigarette, 0 = predicted none

    for r in results:
        img_path = Path(r.path)
        stem = img_path.stem

        # ----- Ground truth: check if label file exists & has content -----
        label_path = Path(labels_dir) / f"{stem}.txt"
        if label_path.exists() and label_path.stat().st_size > 0:
            gt_has_cig = 1
        else:
            gt_has_cig = 0

        # ----- Prediction: if any boxes -> predicted cigarette -----
        pred_has_cig = 1 if (r.boxes is not None and len(r.boxes) > 0) else 0

        y_true.append(gt_has_cig)
        y_pred.append(pred_has_cig)

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    if len(y_true) == 0:
        print("No images found in:", images_dir)
        return

    # ---- Metrics ----
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}  (positive class = cigarette)")
    print(f"Recall   : {rec:.4f}  (positive class = cigarette)")
    print(f"F1-score : {f1:.4f}")

    # ---- Confusion matrix ----
    # rows = GT, cols = prediction
    # labels: 0 = no-cigarette, 1 = cigarette
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\nConfusion matrix (rows = GT, cols = prediction):")
    print(cm)
    print("Row 0: GT = no-cigarette  -> [TN, FP]")
    print("Row 1: GT = cigarette     -> [FN, TP]")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No cigarette", "Cigarette"],
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Cigarette vs No-cigarette)")
    plt.tight_layout()
    plt.show()

    # ---- Bar chart: detected vs missed for positive (cigarette) images ----
    # Only count images where GT = 1
    mask_pos = (y_true == 1)
    pos_true = y_true[mask_pos]
    pos_pred = y_pred[mask_pos]

    if len(pos_true) > 0:
        # On cigarette images: TP = predicted 1, FN = predicted 0
        tp = int(np.sum((pos_true == 1) & (pos_pred == 1)))
        fn = int(np.sum((pos_true == 1) & (pos_pred == 0)))

        labels = ["Detected (TP)", "Missed (FN)"]
        counts = [tp, fn]

        print(f"\nOn cigarette images only:")
        print(f"Detected (TP): {tp}")
        print(f"Missed (FN)  : {fn}")

        plt.figure()
        plt.bar(labels, counts)
        plt.title("Cigarette Images: Detected vs Missed")
        plt.ylabel("Number of images")
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo positive (cigarette) images found in this split for bar chart.")


def main():
    print("Loading model...")
    model = YOLO(WEIGHTS_PATH)

    run_yolo_val(model)
    image_level_metrics_and_confusion(model)


if __name__ == "__main__":
    main()