import os
import random
import shutil
from datetime import datetime
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from ultralytics import YOLO


# ==================== CONFIG (MATCHES YOUR FOLDERS) ====================

# Project base (one level above Dataset)
base_dir = r"C:\Users\jadif\OneDrive\Documentos\Desktop\CV"

# Dataset root
dataset_dir = os.path.join(base_dir, "Dataset")

# Data YAML (change name here if yours is different)
data_yaml_path = os.path.join(dataset_dir, "data.yaml")

# Folder layout:
train_images_dir = os.path.join(dataset_dir, "train", "images")
train_labels_dir = os.path.join(dataset_dir, "train", "labels")
valid_images_dir = os.path.join(dataset_dir, "valid", "images")
valid_labels_dir = os.path.join(dataset_dir, "valid", "labels")
test_images_dir  = os.path.join(dataset_dir, "test",  "images")
test_labels_dir  = os.path.join(dataset_dir, "test",  "labels")

# Where to store final .pt models
model_save_dir = os.path.join(base_dir, "models")

# Preferred YOLO weights (try better model first, fallback to nano)
preferred_yolo_weights = ["yolov8n.pt"]


# ==================== SEEDING ====================

def seed_everything(seed: int = 42):
    import random as pyrandom
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== NOISE / AUGMENT HELPERS ====================

def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 25.0) -> np.ndarray:
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(image: np.ndarray, prob: float = 0.01) -> np.ndarray:
    noisy = image.copy()
    h, w = noisy.shape[:2]
    rnd = np.random.rand(h, w)
    noisy[rnd < prob / 2] = 255
    noisy[rnd > 1 - prob / 2] = 0
    return noisy


def augment_train_with_noise(
    images_dir: str,
    labels_dir: str,
    num_aug: int = 200,
):
    """
    In-place augmentation:
    - Randomly sample num_aug images from train/images
    - For each, create a noisy copy (Gaussian or S&P)
    - Save with suffix _gn or _sp
    - Copy the label file unchanged

    This increases robustness to low-quality CCTV-like images
    without changing box geometry.
    """
    files = [f for f in os.listdir(images_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print("[AUG] No images found, skipping noise augmentation.")
        return

    num_aug = min(num_aug, len(files))
    chosen = random.sample(files, num_aug)
    print(f"[AUG] Creating {num_aug} noisy copies in-place...")

    for f in chosen:
        img_path = os.path.join(images_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            continue

        base, ext = os.path.splitext(f)

        if random.random() < 0.5:
            noisy = add_gaussian_noise(img)
            suffix = "_gn"
        else:
            noisy = add_salt_pepper_noise(img)
            suffix = "_sp"

        new_name = f"{base}{suffix}{ext}"
        new_img_path = os.path.join(images_dir, new_name)
        cv2.imwrite(new_img_path, noisy)

        # copy labels 1:1 (boxes unchanged)
        src_label = os.path.join(labels_dir, base + ".txt")
        dst_label = os.path.join(labels_dir, base + suffix + ".txt")
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

    print("[AUG] Noise augmentation done.")


# ==================== COLOR & HISTOGRAMS ====================

def show_image_and_histograms(img_bgr: np.ndarray, title: str = "image"):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(title)

    plt.subplot(1, 2, 2)
    colors = ("b", "g", "r")
    for i, c in enumerate(colors):
        hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
        plt.plot(hist, label=c)
    plt.legend()
    plt.title("Color histograms")
    plt.tight_layout()
    plt.show()


def random_dataset_histogram(images_dir: str, num_samples: int = 4):
    files = [f for f in os.listdir(images_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print(f"[WARN] No images found in {images_dir}")
        return

    samples = random.sample(files, min(num_samples, len(files)))
    for f in samples:
        path = os.path.join(images_dir, f)
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Failed to read {path}")
            continue
        show_image_and_histograms(img, title=f)


# ==================== EDGE / BLUR QUALITY ====================

def edge_density_score(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 100, 200)
    return edges.mean() / 255.0


def find_really_blurry_images(images_dir: str, threshold: float = 0.02, max_show: int = 5):
    files = [f for f in os.listdir(images_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    bad = []
    for f in files:
        path = os.path.join(images_dir, f)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[WARN] Could not read {path}")
            continue
        score = edge_density_score(gray)
        if score < threshold:
            bad.append((f, score))

    bad = sorted(bad, key=lambda x: x[1])[:max_show]
    for name, s in bad:
        print(f"[BLUR?] {name}: edge_density={s:.4f}")
        img = cv2.imread(os.path.join(images_dir, name))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"{name} (edge_density={s:.4f})")
        plt.axis("off")
        plt.show()


# ==================== YOLO LABEL VISUALIZER ====================

def show_random_labeled_image(images_dir: str, labels_dir: str, class_names=None):
    files = [f for f in os.listdir(images_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print(f"[WARN] No images in {images_dir}")
        return

    name = random.choice(files)
    img_path = os.path.join(images_dir, name)
    label_path = os.path.join(labels_dir, os.path.splitext(name)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = map(float, parts)
                cx *= w
                cy *= h
                bw *= w
                bh *= h
                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2)
                y2 = int(cy + bh / 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if class_names is not None:
                    ci = int(cls_id)
                    if 0 <= ci < len(class_names):
                        label = class_names[ci]
                        cv2.putText(
                            img, label, (x1, max(0, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                        )

    plt.imshow(img)
    plt.title(name)
    plt.axis("off")
    plt.show()


# ==================== SIFT NEAR-DUP CHECK ====================

def sift_similarity(img1_path: str, img2_path: str, ratio_thresh: float = 0.75) -> int:
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return 0

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    return len(good)


def find_near_duplicates(images_dir: str, max_pairs: int = 5, ratio_thresh: float = 0.75, min_good_matches: int = 80):
    files = [f for f in os.listdir(images_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(files) < 2:
        print("[INFO] Not enough images for duplicate checking.")
        return

    pairs_checked = 0
    while pairs_checked < max_pairs:
        f1, f2 = random.sample(files, 2)
        p1 = os.path.join(images_dir, f1)
        p2 = os.path.join(images_dir, f2)
        nm = sift_similarity(p1, p2, ratio_thresh=ratio_thresh)
        if nm >= min_good_matches:
            print(f"[DUP?] {f1} ↔ {f2}  good_matches={nm}")
        pairs_checked += 1


# ==================== YOLO TRAINING ====================

def load_best_yolo_model():
    """Try yolov8s for quality, fall back to yolov8n if something fails."""
    last_err = None
    for w in preferred_yolo_weights:
        try:
            print(f"[INFO] Trying to load {w}...")
            model = YOLO(w)
            print(f"[INFO] Loaded {w} successfully.")
            return model, w
        except Exception as e:
            print(f"[WARN] Failed to load {w}: {e}")
            last_err = e
    raise RuntimeError(f"Could not load any YOLO weights from {preferred_yolo_weights}: {last_err}")


def train_yolo_model(
    epochs: int = 40,
    batch_size: int = 4,
    img_size: int = 512,
    lr0: float = 0.01,
):
    """
    Fine-tune YOLOv8 on the dataset.

    - epochs bumped to 80 for better convergence
    - default model is yolov8s (better than n) with fallback
    """
    seed_everything(42)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"train_{timestamp}"

    model, weight_name = load_best_yolo_model()
    model_short = os.path.splitext(weight_name)[0]  # e.g. "yolov8s"

    print("[INFO] Starting YOLO training...")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=15,            # allow a bit more patience
        save=True,
        device=device,
        project=os.path.join(base_dir, "runs"),
        name=run_name,
        lr0=lr0,
        lrf=0.01,
        plots=True,
        save_period=5,
        # You could tweak augmentations here if needed
    )

    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"{model_short}_{timestamp}.pt")

    print(f"[INFO] Saving trained model to {model_save_path}")
    try:
        model.model.save(model_save_path)
    except AttributeError:
        try:
            model.save(model_save_path)
        except Exception:
            best_model_path = os.path.join(
                base_dir, "runs", run_name, "weights", "best.pt"
            )
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, model_save_path)
                print(f"[INFO] Copied best.pt to {model_save_path}")
            else:
                print("[WARN] Could not find best.pt for fallback copying.")

    return model, run_name, results


# ==================== METRICS PLOTTING ====================

def plot_yolo_results(base_dir: str, run_name: str):
    results_path = os.path.join(base_dir, "runs", run_name, "results.csv")
    if not os.path.exists(results_path):
        print(f"[WARN] No results.csv found at {results_path}")
        return

    epochs = []
    metrics = {}

    with open(results_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(float(row.get("epoch", 0)))
            epochs.append(epoch)
            for k, v in row.items():
                if k == "epoch":
                    continue
                try:
                    val = float(v)
                except (ValueError, TypeError):
                    continue
                metrics.setdefault(k, []).append(val)

    plt.figure()
    if "train/box_loss" in metrics:
        plt.plot(epochs, metrics["train/box_loss"], label="train/box_loss")
    if "train/cls_loss" in metrics:
        plt.plot(epochs, metrics["train/cls_loss"], label="train/cls_loss")
    if "metrics/mAP50(B)" in metrics:
        plt.plot(epochs, metrics["metrics/mAP50(B)"], label="mAP50(B)")
    if "metrics/precision(B)" in metrics:
        plt.plot(epochs, metrics["metrics/precision(B)"], label="precision(B)")
    if "metrics/recall(B)" in metrics:
        plt.plot(epochs, metrics["metrics/recall(B)"], label="recall(B)")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.title("YOLOv8 Training Metrics")
    plt.tight_layout()
    plt.show()


# ==================== EVALUATION (VAL + TEST) ====================

def evaluate_yolo(model: YOLO):
    """
    Use Ultralytics' built-in val() on both valid and test splits.
    Assumes data.yaml has 'val:' (or 'valid:') and 'test:' entries.
    """
    print("=== Evaluating on validation set ===")
    val_metrics = model.val(data=data_yaml_path, split="val")
    # Prints a summary; you can also inspect val_metrics.box.map, etc.

    print("=== Evaluating on test set ===")
    test_metrics = model.val(data=data_yaml_path, split="test")

    return val_metrics, test_metrics


# ==================== PREDICTION VISUALIZATION ====================

def show_yolo_predictions(model: YOLO, images_dir: str, num_samples: int = 4, img_size: int = 512):
    files = [f for f in os.listdir(images_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print(f"[WARN] No images found in {images_dir}")
        return

    num_samples = min(num_samples, len(files))
    chosen = random.sample(files, num_samples)
    paths = [os.path.join(images_dir, f) for f in chosen]

    device = 0 if torch.cuda.is_available() else "cpu"
    results = model.predict(
        paths,
        imgsz=img_size,
        conf=0.25,
        device=device,
        verbose=False,
    )

    plt.figure(figsize=(4 * num_samples, 4))
    for i, (path, res) in enumerate(zip(paths, results)):
        plt.subplot(1, num_samples, i + 1)
        plotted = res.plot()  # BGR
        plotted = plotted[:, :, ::-1]  # BGR -> RGB
        plt.imshow(plotted)
        plt.title(os.path.basename(path))
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ==================== MAIN ====================

def main():
    # ---- 1) Dataset sanity checks ----
    print("=== Histogram sanity check on TRAIN images ===")
    if os.path.isdir(train_images_dir):
        random_dataset_histogram(train_images_dir, num_samples=4)

    print("=== Blur / edge-density check on TRAIN images ===")
    if os.path.isdir(train_images_dir):
        find_really_blurry_images(train_images_dir, threshold=0.02, max_show=5)

    print("=== YOLO label visualization on a random TRAIN image ===")
    if os.path.isdir(train_images_dir) and os.path.isdir(train_labels_dir):
        show_random_labeled_image(train_images_dir, train_labels_dir)

    print("=== SIFT near-duplicate hunt on TRAIN (log only) ===")
    if os.path.isdir(train_images_dir):
        find_near_duplicates(train_images_dir, max_pairs=3, ratio_thresh=0.75, min_good_matches=80)

    # ---- 2) Optional in-place augmentation ----
    print("=== Optional: augment train set with noise ===")
    augment_train_with_noise(train_images_dir, train_labels_dir, num_aug=200)

    # ---- 3) Train YOLO ----
    print("=== Starting YOLO training ===")
    model, run_name, _ = train_yolo_model(
        epochs=40,      # tweak down if you hit time issues
        batch_size=4,
        img_size=512,
        lr0=0.01,
    )
    print(f"[INFO] Training finished. Run name: {run_name}")

    # ---- 4) Plot training curves ----
    print("=== Plotting YOLO training metrics ===")
    plot_yolo_results(base_dir, run_name)

    # ---- 5) Evaluate on val + test ----
    print("=== Evaluating model on val and test splits ===")
    evaluate_yolo(model)

    # ---- 6) Visual predictions ----
    print("=== Showing sample YOLO predictions on VALID images ===")
    if os.path.isdir(valid_images_dir):
        show_yolo_predictions(model, valid_images_dir, num_samples=4, img_size=512)


if __name__ == "__main__":
    main()