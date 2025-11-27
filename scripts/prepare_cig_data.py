
"""
Computer Vision Project – EfficientDet-D0 for Cigarette Detection
-----------------------------------------------------------------

This script merges two YOLO-format datasets into a single-class "cigarette" dataset.

It is intentionally written in a style similar to the course notebooks:
- 09-cnn.ipynb                 (clear sections + simple helper functions)
- 10-trainingCNN.ipynb         (config block + statistics printing)
- 11-pretrainedModels.ipynb    (clean, readable code organization)
- 12-DetectionAndSegmentation.ipynb  (detection-oriented dataset handling)

Final structure created:

Smoking_Combined/
  train/images, train/labels
  val/images,   val/labels
  test/images,  test/labels     (only if the original datasets had test)
  data_combined.yaml            (YOLO-style data config)

Final dataset:
  nc: 1
  names: ['cigarette']
"""

# ===================== IMPORTS (course style) =====================
# Similar to how we import standard libraries in 09-cnn.ipynb and 10-trainingCNN.ipynb
import os
from glob import glob
import shutil
from collections import defaultdict
from argparse import Namespace
import textwrap


# ===================== CONFIG (inspired by 10-trainingCNN.ipynb) =====================
# Using Namespace for configuration, like we used for hyperparameters there.

cfg = Namespace(
    # TODO: CHANGE THESE PATHS TO MATCH YOUR MACHINE
    dataset1_root = r"C:\Users\desk-14\Desktop\Computer_Vision\Datasets\Dataset_1",   # dataset with names: ['drinking', 'smoking']
    dataset2_root = r"C:\Users\desk-14\Desktop\Computer_Vision\Datasets\Dataset_2",   # dataset with names: ['Cigarette','Person','Smoke','Vape','smoking']
    out_root      = r"C:\Users\desk-14\Desktop\Computer_Vision\Datasets\Smoking_Combined",

    # Unified classes: 0 -> cigarette
    # Dataset 1: names: ['drinking', 'smoking']
    #   - keep ONLY class 1 ('smoking') and treat it as "cigarette"
    dataset1_class_map = {
        1: 0,  # smoking -> cigarette
        # class 0 (drinking) is dropped
    },

    # Dataset 2: names: ['Cigarette','Person','Smoke','Vape','smoking']
    #   - keep ONLY class 0 ('Cigarette') as "cigarette"
    dataset2_class_map = {
        0: 0,  # Cigarette -> cigarette
        # 1: Person  -> dropped
        # 2: Smoke   -> dropped
        # 3: Vape    -> dropped
        # 4: smoking -> dropped (we only care about explicit cigarette boxes)
    },

    final_class_names = ['cigarette'],

    # Expected image extensions and split names
    image_exts = [".jpg", ".jpeg", ".png"],
    split_map = {
        "train": "train",
        "valid": "val",   # Roboflow often uses 'valid'
        "val":   "val",
        "test":  "test",
    }
)


# ===================== HELPER FUNCTIONS =====================
# These are written in the same spirit as the small utilities we used
# in 02-color.ipynb and 03-filtering.ipynb (simple, focused helpers).


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def find_image(images_dir: str, base_name: str):
    """
    Find an image file with given base name and a known extension.

    This is similar in spirit to how we handled image paths in 02-color.ipynb,
    where we systematically loaded images from disk.
    """
    for ext in cfg.image_exts:
        candidate = os.path.join(images_dir, base_name + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def process_dataset(dataset_root: str,
                    class_map: dict,
                    prefix: str,
                    counters: dict) -> None:
    """
    Copy and remap a YOLO-format dataset into cfg.out_root.

    - dataset_root: root path of source dataset (with train/val/test subfolders)
    - class_map: mapping from original class id -> new class id (0 for cigarette)
    - prefix: 'd1' or 'd2' to avoid filename collisions
    - counters: dictionary to accumulate stats

    The idea of collecting stats in a 'counters' dict is inspired by the way
    we tracked losses/accuracies per epoch in 10-trainingCNN.ipynb.
    """
    for src_split, tgt_split in cfg.split_map.items():
        # labels dir
        labels_dir = os.path.join(dataset_root, src_split, "labels")
        if not os.path.isdir(labels_dir):
            # If this split does not exist in this dataset, skip
            continue

        # images dir
        images_dir = os.path.join(dataset_root, src_split, "images")
        if not os.path.isdir(images_dir):
            print(f"[WARN] No images dir for split '{src_split}' in {dataset_root}")
            continue

        # output dirs
        out_images_dir = os.path.join(cfg.out_root, tgt_split, "images")
        out_labels_dir = os.path.join(cfg.out_root, tgt_split, "labels")
        ensure_dir(out_images_dir)
        ensure_dir(out_labels_dir)

        label_files = glob(os.path.join(labels_dir, "*.txt"))
        print(f"[INFO] {prefix} - {src_split} -> {tgt_split}: {len(label_files)} label files")

        for label_path in label_files:
            base = os.path.splitext(os.path.basename(label_path))[0]

            img_path = find_image(images_dir, base)
            if img_path is None:
                print(f"[WARN] No image for label: {label_path}")
                counters["missing_image"] += 1
                continue

            # New base name to avoid collisions between dataset 1 & 2.
            # This is similar in spirit to how we created unique experiment
            # names/checkpoints in 10-trainingCNN.ipynb.
            new_base = f"{prefix}_{src_split}_{base}"
            img_ext = os.path.splitext(img_path)[1]
            new_img_path = os.path.join(out_images_dir, new_base + img_ext)
            new_label_path = os.path.join(out_labels_dir, new_base + ".txt")

            # Copy image file
            shutil.copy2(img_path, new_img_path)

            new_lines = []
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    try:
                        cid = int(parts[0])
                    except ValueError:
                        print(f"[WARN] Bad line in {label_path}: {line}")
                        counters["bad_lines"] += 1
                        continue

                    # Only keep classes that map to "cigarette"
                    if cid not in class_map:
                        # e.g. drinking, Person, Smoke, Vape, generic 'smoking' (4)
                        continue

                    new_cid = class_map[cid]  # always 0 in our mapping
                    parts[0] = str(new_cid)
                    new_lines.append(" ".join(parts))
                    counters["boxes_total"] += 1

            if not new_lines:
                # No cigarette boxes left in this image -> drop it
                os.remove(new_img_path)
                counters["images_dropped_no_boxes"] += 1
            else:
                counters["images_total"] += 1
                with open(new_label_path, "w") as f_out:
                    f_out.write("\n".join(new_lines))


def write_yaml(out_root: str, class_names: list) -> None:
    """
    Write a YOLO-style data YAML for the combined dataset.

    This mirrors the kind of small config files we implicitly used in
    12-DetectionAndSegmentation.ipynb when we loaded YOLO models with paths.
    """
    yaml_path = os.path.join(out_root, "data_combined.yaml")
    content = textwrap.dedent(f"""\
        train: ./train/images
        val: ./val/images
        test: ./test/images

        nc: {len(class_names)}
        names: {class_names}
    """)
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"[INFO] Wrote YAML: {yaml_path}")


# ===================== MAIN (same idea as training entry point) =====================

def main():
    # The overall flow here (prepare data, then print a summary)
    # is conceptually similar to the training scripts in 10-trainingCNN.ipynb,
    # where we set up everything and then call Trainer.fit().

    if os.path.exists(cfg.out_root):
        print(f"[INFO] Output dir: {cfg.out_root}")
    ensure_dir(cfg.out_root)

    counters = defaultdict(int)

    print("=== Processing Dataset 1 (['drinking','smoking']) ===")
    process_dataset(cfg.dataset1_root, cfg.dataset1_class_map, prefix="d1", counters=counters)

    print("=== Processing Dataset 2 (['Cigarette','Person','Smoke','Vape','smoking']) ===")
    process_dataset(cfg.dataset2_root, cfg.dataset2_class_map, prefix="d2", counters=counters)

    print("\n=== SUMMARY (similar style to training logs in 10-trainingCNN.ipynb) ===")
    print(f"Images kept (with cigarette boxes): {counters['images_total']}")
    print(f"Images dropped (no cigarette boxes): {counters['images_dropped_no_boxes']}")
    print(f"Images with missing image file: {counters['missing_image']}")
    print(f"Bad annotation lines skipped: {counters['bad_lines']}")
    print(f"Total cigarette boxes: {counters['boxes_total']}")

    write_yaml(cfg.out_root, cfg.final_class_names)
    print("\n[DONE] Combined 'cigarette' dataset is ready.")


if __name__ == "__main__":
    # Standard Python entry point, like we would use if we converted
    # 10-trainingCNN.ipynb to a .py training script.
    main()
