# EyePACS Retina Image Quality Classification

This repository contains a deep learning pipeline for classifying retinal fundus image quality using the EyeQ / EyePACS dataset.

The project focuses on preparing retinal images, creating clean dataset splits, training a ConvNeXt-Tiny model, and analyzing model errors after training.

---

## Project Overview

Retinal fundus images are commonly used in medical AI tasks such as diabetic retinopathy detection and eye disease screening.
However, image quality has a direct impact on model performance.

This project aims to classify retinal images into quality categories so that low-quality images can be identified before being used in downstream medical diagnosis models.

---

## Quality Classes

| Label | Class Name | Description                                |
| ----: | ---------- | ------------------------------------------ |
|     0 | Good       | High-quality retinal image                 |
|     1 | Usable     | Acceptable image with minor quality issues |
|     2 | Reject     | Low-quality image that should not be used  |

---

## Current Project Structure

```text
EyePACS_Model1_Clean/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── convnext_tiny_512_run1/
│   └── Training outputs, logs, plots, and model results
│
├── EyeQ_Preprocessed_512/
│   └── Preprocessed 512x512 retinal images
│
├── analyze_eyeq_model_errors.py
├── check_gpu_and_convnext_architecture.py
├── create_eyeq_master_split.py
├── prepare_eyeq_dataset.py
├── preprocess_eyeq_512.py
├── profile_eyeq_split_images.py
└── train_convnext_tiny_eyeq_512.py
```

---

## Main Files Description

### `prepare_eyeq_dataset.py`

Prepares the EyeQ dataset and organizes the required metadata or image paths before preprocessing and splitting.

---

### `preprocess_eyeq_512.py`

Preprocesses retinal images and resizes them to **512x512**.

The output folder is:

```text
EyeQ_Preprocessed_512/
```

---

### `create_eyeq_master_split.py`

Creates the final master split for training, validation, and testing.

This script is responsible for organizing the dataset into clean splits before model training.

---

### `profile_eyeq_split_images.py`

Profiles the dataset splits and checks important dataset statistics such as:

* Number of images.
* Class distribution.
* Train/validation/test balance.
* Image availability.

---

### `check_gpu_and_convnext_architecture.py`

Checks the available GPU environment and verifies the ConvNeXt-Tiny architecture before training.

This helps confirm that the training environment is ready.

---

### `train_convnext_tiny_eyeq_512.py`

Main training script for the ConvNeXt-Tiny model using 512x512 retinal images.

This script handles:

* Loading the dataset.
* Applying transformations.
* Loading ConvNeXt-Tiny.
* Training the model.
* Validating the model.
* Saving results.

---

### `analyze_eyeq_model_errors.py`

Analyzes model mistakes after training.

This script can be used to inspect:

* Misclassified images.
* Class-level errors.
* Confusion patterns.
* Weaknesses in model predictions.

---

## Experiment Folder

### `convnext_tiny_512_run1/`

This folder contains outputs related to the first ConvNeXt-Tiny experiment.

It may include:

* Training logs.
* Evaluation reports.
* Confusion matrix.
* Loss and accuracy plots.
* Saved model weights.
* Error analysis outputs.

Important note: large files such as model weights should not be pushed to GitHub.

---

## Dataset Folder

### `EyeQ_Preprocessed_512/`

This folder contains the preprocessed retinal images resized to 512x512.

This folder should remain local and should not be uploaded to GitHub because image datasets are usually large.

Expected local path:

```text
C:\Users\anwar\OneDrive\Desktop\EyePACS_Model1_Clean\EyeQ_Preprocessed_512
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/retina-image-quality-classification.git
cd retina-image-quality-classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

## Recommended Workflow

Run the project in this order:

### 1. Prepare Dataset

```bash
python prepare_eyeq_dataset.py
```

### 2. Preprocess Images to 512x512

```bash
python preprocess_eyeq_512.py
```

### 3. Create Master Split

```bash
python create_eyeq_master_split.py
```

### 4. Profile Dataset Splits

```bash
python profile_eyeq_split_images.py
```

### 5. Check GPU and Model Architecture

```bash
python check_gpu_and_convnext_architecture.py
```

### 6. Train ConvNeXt-Tiny Model

```bash
python train_convnext_tiny_eyeq_512.py
```

### 7. Analyze Model Errors

```bash
python analyze_eyeq_model_errors.py
```

---

## Training Pipeline

The training process is based on ConvNeXt-Tiny with 512x512 retinal images.

Main training steps:

* Load preprocessed images.
* Load train/validation/test split.
* Apply image transformations.
* Load ConvNeXt-Tiny architecture.
* Train the model.
* Validate model performance.
* Save the best model.
* Generate performance results.

---

## Evaluation

The model should be evaluated using:

* Accuracy.
* Precision.
* Recall.
* Macro F1-score.
* Confusion matrix.
* Classification report.
* Error analysis.

The most important metric for this project is **Macro F1-score**, because the dataset may have class imbalance between Good, Usable, and Reject classes.

---

## GitHub Team Workflow

Team members should not push directly to the `main` branch.

Recommended branches:

```text
main        Stable version
dev         Development branch
feature/*   New feature
fix/*       Bug fix
docs/*      Documentation update
experiment/* Model experiment
```

Example:

```bash
git switch -c feature/error-analysis
git add .
git commit -m "feat: add model error analysis script"
git push origin feature/error-analysis
```

After pushing, open a Pull Request on GitHub.

---

## Suggested GitHub Tasks

### Task 1: Repository Setup

* Add README.md.
* Add `.gitignore`.
* Add requirements.txt.
* Organize current scripts.
* Create GitHub Project Board.

---

### Task 2: Dataset Preparation Review

* Review `prepare_eyeq_dataset.py`.
* Check if all image paths are correct.
* Confirm label mapping.
* Document dataset source and folder structure.

---

### Task 3: Image Preprocessing

* Review `preprocess_eyeq_512.py`.
* Confirm output image size is 512x512.
* Check number of processed images.
* Save preprocessing summary.

---

### Task 4: Master Split Creation

* Review `create_eyeq_master_split.py`.
* Confirm train/validation/test split.
* Check class distribution in each split.
* Prevent data leakage.

---

### Task 5: Dataset Profiling

* Review `profile_eyeq_split_images.py`.
* Generate class distribution report.
* Check missing images.
* Save dataset profile results.

---

### Task 6: GPU and Architecture Check

* Review `check_gpu_and_convnext_architecture.py`.
* Confirm GPU availability.
* Confirm ConvNeXt-Tiny model structure.
* Confirm classifier output matches 3 classes.

---

### Task 7: Model Training

* Review `train_convnext_tiny_eyeq_512.py`.
* Train ConvNeXt-Tiny model.
* Save best model.
* Save training history.
* Save plots and metrics.

---

### Task 8: Error Analysis

* Review `analyze_eyeq_model_errors.py`.
* Analyze misclassified images.
* Identify weak classes.
* Save error samples.
* Write conclusions.

---

### Task 9: Documentation

* Keep README updated.
* Add script usage instructions.
* Add experiment notes.
* Add results summary.
* Document team workflow.

---

## Suggested Project Board

Use these columns in GitHub Projects:

```text
Backlog
To Do
In Progress
Review
Done
```

---

## Suggested Labels

```text
type: feature
type: bug
type: docs
type: experiment
area: data
area: preprocessing
area: training
area: evaluation
priority: high
priority: medium
priority: low
blocked
```

---

## Commit Message Convention

Use clear commit messages:

```text
feat: add dataset preprocessing script
feat: create master dataset split
feat: train convnext tiny model
fix: correct image path issue
docs: update README instructions
chore: add gitignore file
refactor: clean training script
```

---

## Important Notes

* Do not upload raw or preprocessed image folders to GitHub.
* Do not upload large model weights such as `.pth`, `.pt`, or `.ckpt`.
* Do not push directly to `main`.
* Every task should have its own branch.
* Every branch should be merged using Pull Request.
* Dataset validation should be completed before training.
* Keep experiment outputs organized inside experiment folders.

---

## Files That Should Not Be Uploaded

The following should stay local:

```text
EyeQ_Preprocessed_512/
*.pth
*.pt
*.ckpt
*.h5
*.onnx
large output files
raw image folders
```

---

## Current Status

Project status: **Model 1 clean experiment**

Current experiment:

```text
convnext_tiny_512_run1
```

Current model:

```text
ConvNeXt-Tiny
```

Current image size:

```text
512x512
```

---

## Future Improvements

* Move scripts into organized folders such as `src/`, `scripts/`, and `configs/`.
* Add configuration file for training parameters.
* Add experiment tracking.
* Add more model architectures for comparison.
* Add better error analysis visualization.
* Add automated tests for dataset loading.
* Add final report summarizing results.

---

## License

This project is for academic and research purposes.
