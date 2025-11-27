# cv-smoking-knife-detection
AUS Computer Vision Final Project, cigarette&amp; knife detection using YOLO
HOW TO RUN THE CODE (TRAINING AND TESTING)

This README explains, step by step, how to run the code for training and testing
the YOLOv8n detectors for cigarettes and knives for this course project.
Before running any script, make sure Python is installed and that you have
installed all packages listed in requirements.txt.

------------------------------------------------------------
1. TRAINING – CIGARETTE DETECTOR (YOLOv8n)
------------------------------------------------------------

Step 1: (Optional) Rebuild the merged cigarette dataset

If you need to merge and clean two separate cigarette datasets, edit the file
prepare_cigarette_dataset.py in the cfg = Namespace(...) section so that:

- dataset1_root points to your first cigarette dataset (YOLO format).
- dataset2_root points to your second cigarette dataset (YOLO format).
- out_root points to the folder where you want the merged dataset to be created.

Then run from the project folder that contains prepare_cigarette_dataset.py:

   python prepare_cigarette_dataset.py

This script creates a merged YOLO dataset under out_root, with train, val, and test
folders and a data_combined.yaml file describing the dataset.
The result is a single-class cigarette dataset.

Step 2: Point CigDetector.py to the correct paths

Open CigDetector.py and in the CONFIG section at the top:

- Set base_dir to the folder on your machine where the cigarette Dataset folder
  and the YOLO data yaml file are located.
- Confirm that base_dir\Dataset contains:
    train/images
    train/labels
    valid/images (or val/images)
    valid/labels (or val/labels)
    test/images
    test/labels
- Make sure data_yaml_path points to the correct YOLO data yaml file
  (for example data.yaml or data_combined.yaml) with fields:
  train, val, test, nc, and names.

Step 3: Run cigarette training

From the project folder that contains CigDetector.py, run:

   python CigDetector.py

The script will do the following, in order:

- Run basic sanity checks on the training images:
  color histograms, blur checks, label visualization, near-duplicate checks.
- Optionally augment the training set by creating noisy copies of some images
  and copying their labels.
- Train a YOLOv8n model on the cigarette dataset using the ultralytics YOLO API.
- Save the trained model checkpoint (a .pt file) into the models folder under base_dir.
- Evaluate the model on the validation and test splits using YOLO's built-in val method.
- Show example predictions on a few images from the validation split.

------------------------------------------------------------
2. TRAINING – KNIFE DETECTOR (YOLOv8n)
------------------------------------------------------------

Step 1: Place the knife dataset in YOLOv8 format

Choose a project folder for the knife detector and set it as base_dir.
Inside that folder, create the following structure:

   base_dir
     Dataset
       train/images
       train/labels
       valid/images
       valid/labels
       test/images
       test/labels
       data.yaml

The data.yaml file must describe the knife dataset and contain the paths to
train, val, and test, plus nc and names. For a single knife class, names
will normally be ['knife'] and nc will be 1.

Step 2: Point KnifeDetector.py to the correct paths

Open KnifeDetector.py and in the CONFIG section at the top:

- Set base_dir to your knife project folder (the folder that contains Dataset).
- Make sure data_yaml_path points to Dataset\data.yaml or to the correct
  YOLO data yaml file for the knife dataset.

Step 3: Run knife training

From the project folder that contains KnifeDetector.py, run:

   python KnifeDetector.py

The script will perform the same sequence of steps as CigDetector.py, but on
the knife dataset:

- Dataset sanity checks (histograms, blur check, label visualization, near-duplicate checks).
- Optional noise-based augmentation of the training set.
- Training a YOLOv8n knife detector.
- Saving the trained .pt model into the models folder under base_dir.
- Evaluation on validation and test splits via YOLO's val method.
- Displaying sample predictions on validation images.

------------------------------------------------------------
3. TESTING AND EVALUATION – BOX-LEVEL METRICS
------------------------------------------------------------

For detailed testing and evaluation of a trained model (precision, recall,
F1-score, and a confusion matrix at the bounding-box level), use Evaluation.py.

Step 1: Configure Evaluation.py

Open Evaluation.py and locate the CONFIG section for the detector you want
to evaluate (cigarette or knife). Then:

- Set the class ID (for example CIG_CLASS_ID or KNIFE_CLASS_ID) so that it
  matches the index of the target class in your YOLO labels.
- Set BASE_DIR to the project folder that contains the Dataset for this detector.
- Set WEIGHTS_PATH to the path of the trained model .pt file that you want
  to evaluate. This can be one of the files saved in the models folder or the
  best.pt file from a specific training run.
- Ensure DATASET_ROOT is set correctly, usually as BASE_DIR joined with "Dataset".
- Set SPLIT to "test" (or "valid"/"val") to choose which split to evaluate.

Step 2: Run evaluation

From the project folder that contains Evaluation.py, run:

   python Evaluation.py

The script will:

- Run YOLOv8 inference on all images in the chosen split and save prediction
  txt files and annotated images in a prediction output folder.
- For the selected class, match predicted boxes to ground-truth boxes using
  an IoU threshold.
- Accumulate global counts of true positives (TP), false positives (FP),
  and false negatives (FN).
- Compute precision, recall, and F1-score from these counts.
- Build and display a confusion matrix and a bar chart summarizing how many
  ground-truth boxes were detected versus missed.

These steps complete the instructions needed to run the code for both training
and testing the YOLOv8n cigarette and knife detectors for this course project.
