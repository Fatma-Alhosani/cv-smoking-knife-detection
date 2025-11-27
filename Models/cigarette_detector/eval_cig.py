"""
evaluate_yolo_cigarette_boxes.py

BOX-LEVEL EVALUATION FOR CIGARETTE DETECTION (with confusion matrix + bar chart)

Pipeline:
1) Run YOLOv8 on the chosen split (val/test) and save predictions as YOLO .txt files
   AND annotated images.
2) Compare prediction .txt files with GT .txt files at the BOX level, for a single class:
       CIG_CLASS_ID  (usually 0 for 'cigarette')

Definitions (box-level):

- GT boxes: all boxes in GT label files with class == CIG_CLASS_ID
- Pred boxes: all boxes in prediction txt files with class == CIG_CLASS_ID

- A predicted box is a TRUE POSITIVE (TP) if:
    * It has class == CIG_CLASS_ID, and
    * It matches a GT box of the same class with IoU >= IOU_MATCH_THRESH,
    * And that GT box is not already matched to another prediction (greedy matching).

- A predicted box is a FALSE POSITIVE (FP) if:
    * It does NOT match any GT box (of the same class) with IoU >= IOU_MATCH_THRESH.

- A GT box is a FALSE NEGATIVE (FN) if:
    * No prediction matches it with IoU >= IOU_MATCH_THRESH.

Metrics:

- Precision = TP / (TP + FP)
- Recall    = TP / (TP + FN)
- F1-score  = harmonic mean of precision & recall

Confusion matrix (box-level, for reporting only):

    [[ TN , FP ],
     [ FN , TP ]]

We do NOT enumerate background negatives, so we set:
    TN = 0
and interpret:

- Row 0 ("No GT box"): [TN, FP]  -> FP are predicted boxes with no matching GT
- Row 1 ("GT box"):    [FN, TP]  -> TP/FN are GT boxes that are detected / missed

Bar chart:
- "Detected (TP)" vs "Missed (FN)" for GT cigarette boxes.

REQUIREMENTS (in your venv):
    pip install ultralytics numpy matplotlib scikit-learn
"""

import os
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from ultralytics import YOLO


# ===================== CONFIG – EDIT THESE IF NEEDED =====================

# Index of "cigarette" in your YOLO labels (if names: ['cigarette', 'human'], this is 0)
CIG_CLASS_ID = 0

# 1) Trained weights
WEIGHTS_PATH = r"C:\Users\lenovo\Desktop\Cig\runs\train_20251126_231444\weights\best.pt"

# 2) Dataset root + which split to evaluate on ("val" or "test")
DATASET_ROOT = r"C:\Users\lenovo\Desktop\Cig\Dataset"
SPLIT = "test"  # "val" or "test"

# 3) Inference settings
IMG_SIZE = 512
CONF_THRESH = 0.25
IOU_PREDICT = 0.5  # IoU for NMS in YOLO inference (NOT the box-match IoU)

# 4) Box-match IoU threshold (for counting TP vs FP/FN)
IOU_MATCH_THRESH = 0.5

# 5) Where to store prediction txt files and annotated images (Ultralytics project/name)
PRED_PROJECT_DIR = os.path.join(DATASET_ROOT, "predictions")
PRED_RUN_NAME = f"pred_boxes_{SPLIT}"  # folder name inside PRED_PROJECT_DIR

# ========================================================================


def cxcywh_to_xyxy(box):
    """
    Convert a YOLO-format box [cx, cy, w, h] (normalized 0–1)
    to [x1, y1, x2, y2] (also 0–1).
    """
    cx, cy, w, h = box
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return [x1, y1, x2, y2]


def load_yolo_boxes(txt_path: Path, target_class_id: int):
    """
    Load YOLO boxes of a specific class from a txt file.

    Format per line:
        class cx cy w h [conf]

    Returns:
        boxes_xyxy: list of [x1, y1, x2, y2], all floats in [0,1]
    """
    boxes_xyxy = []

    if not (txt_path.exists() and txt_path.stat().st_size > 0):
        return boxes_xyxy

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                continue

            if cls_id != target_class_id:
                continue

            try:
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            xyxy = cxcywh_to_xyxy([cx, cy, w, h])
            boxes_xyxy.append(xyxy)

    return boxes_xyxy


def iou_matrix(gt_boxes, pred_boxes):
    """
    Compute IoU matrix between GT boxes and predicted boxes.

    gt_boxes   : list of [x1, y1, x2, y2]
    pred_boxes : list of [x1, y1, x2, y2]

    Returns:
        IoU matrix of shape (len(gt_boxes), len(pred_boxes))
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)

    gt = np.array(gt_boxes)   # [N, 4]
    pr = np.array(pred_boxes) # [M, 4]

    # gt: [N,1,4], pr: [1,M,4] -> broadcast to [N,M,4]
    gt = gt[:, None, :]
    pr = pr[None, :, :]

    # Intersection coords
    inter_x1 = np.maximum(gt[..., 0], pr[..., 0])
    inter_y1 = np.maximum(gt[..., 1], pr[..., 1])
    inter_x2 = np.minimum(gt[..., 2], pr[..., 2])
    inter_y2 = np.minimum(gt[..., 3], pr[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter_area = inter_w * inter_h

    # Areas
    gt_area = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
    pr_area = (pr[..., 2] - pr[..., 0]) * (pr[..., 3] - pr[..., 1])

    union_area = gt_area + pr_area - inter_area
    iou = np.zeros_like(inter_area)
    valid = union_area > 0
    iou[valid] = inter_area[valid] / union_area[valid]

    return iou  # shape (N, M)


def match_boxes_per_image(gt_boxes, pred_boxes, iou_thresh=0.5):
    """
    Given GT boxes and predicted boxes for ONE image:

    - Greedy match based on IoU (always pick highest IoU remaining)
    - A match is accepted if IoU >= iou_thresh
    - Each GT and each pred can be matched at most once.

    Returns:
        tp: number of true positive boxes
        fp: number of false positive boxes
        fn: number of false negative boxes
    """
    num_gt = len(gt_boxes)
    num_pr = len(pred_boxes)

    if num_gt == 0 and num_pr == 0:
        return 0, 0, 0
    if num_gt == 0 and num_pr > 0:
        # No GT boxes, all predictions are FP
        return 0, num_pr, 0
    if num_gt > 0 and num_pr == 0:
        # GT exists but no predictions -> all GT are FN
        return 0, 0, num_gt

    iou_mat = iou_matrix(gt_boxes, pred_boxes)  # [num_gt, num_pr]

    # Greedy matching
    matched_gt = set()
    matched_pr = set()
    tp = 0

    while True:
        # Find the highest IoU pair
        max_iou = 0.0
        max_gt = -1
        max_pr = -1

        for gi in range(num_gt):
            if gi in matched_gt:
                continue
            for pi in range(num_pr):
                if pi in matched_pr:
                    continue
                iou_val = iou_mat[gi, pi]
                if iou_val > max_iou:
                    max_iou = iou_val
                    max_gt = gi
                    max_pr = pi

        if max_iou < iou_thresh or max_gt == -1 or max_pr == -1:
            break

        # Accept this match
        matched_gt.add(max_gt)
        matched_pr.add(max_pr)
        tp += 1

    fp = num_pr - len(matched_pr)
    fn = num_gt - len(matched_gt)

    return tp, fp, fn


def run_predictions_and_save_txt(model: YOLO):
    """
    Run YOLO on all images in the chosen SPLIT and save predictions as YOLO txt files
    AND annotated images.

    Ultralytics will create:
        PRED_PROJECT_DIR / PRED_RUN_NAME / labels / <image_stem>.txt
        PRED_PROJECT_DIR / PRED_RUN_NAME / <annotated_images>.jpg

    We FIRST CLEAR the old prediction folder so old txts don't pollute the comparison.

    Returns:
        labels_dir (str): path to GT labels dir
        pred_labels_dir (str): path to prediction labels dir
        image_filenames (list[str]): list of image filenames in the split
    """
    images_dir = os.path.join(DATASET_ROOT, SPLIT, "images")
    labels_dir = os.path.join(DATASET_ROOT, SPLIT, "labels")

    print("\n========== RUNNING PREDICTIONS & SAVING TXT + IMAGES (BOX-LEVEL) ==========")
    print(f"Images dir       : {images_dir}")
    print(f"GT labels dir    : {labels_dir}")
    print(f"Pred project dir : {PRED_PROJECT_DIR}")
    print(f"Pred run name    : {PRED_RUN_NAME}")

    pred_run_dir = os.path.join(PRED_PROJECT_DIR, PRED_RUN_NAME)
    if os.path.isdir(pred_run_dir):
        print(f"Removing old prediction folder: {pred_run_dir}")
        shutil.rmtree(pred_run_dir, ignore_errors=True)

    # Run YOLO inference, saving txt predictions AND annotated images
    model.predict(
        source=images_dir,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        iou=IOU_PREDICT,
        save=True,           # <--- SAVE ANNOTATED IMAGES NOW
        save_txt=True,       # save YOLO-format txt predictions
        save_conf=False,     # keep it simple: class cx cy w h
        project=PRED_PROJECT_DIR,
        name=PRED_RUN_NAME,
        exist_ok=True,
        verbose=False,
    )

    pred_labels_dir = os.path.join(pred_run_dir, "labels")
    print(f"Prediction txt dir   : {pred_labels_dir}")
    print(f"Annotated images dir : {pred_run_dir}")

    # Collect list of image filenames
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images_dir_p = Path(images_dir)
    image_filenames = [
        f.name for f in images_dir_p.iterdir()
        if f.suffix.lower() in exts
    ]
    image_filenames.sort()

    print(f"Number of images in split '{SPLIT}': {len(image_filenames)}")

    return labels_dir, pred_labels_dir, image_filenames


def evaluate_box_level(labels_dir: str, pred_labels_dir: str, image_filenames):
    """
    For each image:
      - Load GT boxes (cigarette class) from labels_dir
      - Load predicted boxes (cigarette class) from pred_labels_dir
      - Match with IoU >= IOU_MATCH_THRESH (greedy)
      - Sum TP, FP, FN over all images

    Then compute global precision, recall, F1 at the BOX level,
    and show a confusion matrix + bar chart.
    """
    print("\n========== BOX-LEVEL EVALUATION (cigarette class only) ==========")
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_name in image_filenames:
        stem = Path(img_name).stem

        gt_txt_path = Path(labels_dir) / f"{stem}.txt"
        pred_txt_path = Path(pred_labels_dir) / f"{stem}.txt"

        gt_boxes = load_yolo_boxes(gt_txt_path, CIG_CLASS_ID)
        pred_boxes = load_yolo_boxes(pred_txt_path, CIG_CLASS_ID)

        tp, fp, fn = match_boxes_per_image(gt_boxes, pred_boxes, iou_thresh=IOU_MATCH_THRESH)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    print(f"\nTotal GT boxes (cigarette): {total_tp + total_fn}")
    print(f"Total predicted boxes (cigarette): {total_tp + total_fp}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")

    # Avoid division by zero
    if (total_tp + total_fp) > 0:
        precision = total_tp / (total_tp + total_fp)
    else:
        precision = 0.0

    if (total_tp + total_fn) > 0:
        recall = total_tp / (total_tp + total_fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    print(f"\nIoU match threshold : {IOU_MATCH_THRESH:.2f}")
    print(f"Precision (box-level): {precision:.4f}")
    print(f"Recall    (box-level): {recall:.4f}")
    print(f"F1-score  (box-level): {f1:.4f}")

    # ================== BOX-LEVEL "CONFUSION MATRIX" ==================
    cm = np.array([[0,           total_fp],
                   [total_fn,    total_tp]], dtype=int)

    print("\nBox-level confusion-style matrix (rows = GT, cols = prediction):")
    print(cm)
    print("Row 0: GT = No GT box     -> [TN (0), FP]")
    print("Row 1: GT = GT cigarette  -> [FN, TP]")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No GT box", "GT cigarette"],
    )
    disp.plot(cmap="Blues")
    plt.title("Box-level Confusion (Cigarette vs No-GT-Match)")
    plt.tight_layout()
    plt.show()

    # ================== BAR CHART: TP vs FN (GT boxes) ==================
    labels = ["Detected (TP)", "Missed (FN)"]
    counts = [total_tp, total_fn]

    plt.figure()
    plt.bar(labels, counts)
    plt.title("GT Cigarette Boxes: Detected vs Missed")
    plt.ylabel("Number of boxes")
    plt.tight_layout()
    plt.show()


# ========================================================================

def main():
    print("Loading model...")
    model = YOLO(WEIGHTS_PATH)

    # 1) Run predictions and save txts + annotated images
    labels_dir, pred_labels_dir, image_filenames = run_predictions_and_save_txt(model)

    # 2) Box-level evaluation with confusion matrix + bar chart
    evaluate_box_level(labels_dir, pred_labels_dir, image_filenames)


if __name__ == "__main__":
    main()
