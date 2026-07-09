from pathlib import Path
import time
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight


# هذا السكربت يدرب نموذج ConvNeXt-Tiny لتصنيف جودة صور الشبكية EyeQ إلى 3 فئات.
# يستخدم صورا مجهزة مسبقا بحجم 512x512 ولا يعيد تغيير حجمها داخل التدريب.
# يستخدم class weights لتقليل تأثير عدم توازن الفئات.
# يراقب مقاييس التدريب والتحقق في كل epoch ويولد رسومات تساعد على كشف overfitting و underfitting.
# يحفظ أفضل نموذج حسب validation Macro F1-score، ويقيم النموذج الأفضل على test set.
# لا يقوم هذا السكربت بتعديل ملفات البيانات أو حذفها أو نقلها أو إعادة تسميتها.


DATASET_ROOT = Path(r"C:\EyePACS\EyeQ_Preprocessed_512")

TRAIN_IMAGE_DIR = DATASET_ROOT / "train"
VAL_IMAGE_DIR = DATASET_ROOT / "val"
TEST_IMAGE_DIR = DATASET_ROOT / "test"

INPUT_REPORTS_DIR = DATASET_ROOT / "reports"
TRAIN_CSV_PATH = INPUT_REPORTS_DIR / "train_preprocessed.csv"
VAL_CSV_PATH = INPUT_REPORTS_DIR / "val_preprocessed.csv"
TEST_CSV_PATH = INPUT_REPORTS_DIR / "test_preprocessed.csv"

RUN_DIR = Path(r"C:\EyePACS\training_runs\convnext_tiny_512_run1")
CHECKPOINTS_DIR = RUN_DIR / "checkpoints"
REPORTS_DIR = RUN_DIR / "reports"
PLOTS_DIR = RUN_DIR / "plots"

BEST_F1_MODEL_PATH = CHECKPOINTS_DIR / "best_model_by_macro_f1.pth"
BEST_LOSS_MODEL_PATH = CHECKPOINTS_DIR / "best_model_by_val_loss.pth"
LAST_MODEL_PATH = CHECKPOINTS_DIR / "last_model.pth"

EPOCH_METRICS_PATH = REPORTS_DIR / "epoch_metrics.csv"
FINAL_REPORT_PATH = REPORTS_DIR / "final_training_report.txt"
VAL_CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "validation_classification_report.txt"
TEST_CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "test_classification_report.txt"
TEST_METRICS_SUMMARY_PATH = REPORTS_DIR / "test_metrics_summary.csv"
CLASS_WEIGHTS_PATH = REPORTS_DIR / "class_weights.csv"

LOSS_CURVE_PATH = PLOTS_DIR / "loss_curve.png"
ACCURACY_CURVE_PATH = PLOTS_DIR / "accuracy_curve.png"
MACRO_F1_CURVE_PATH = PLOTS_DIR / "macro_f1_curve.png"
PRECISION_RECALL_CURVE_PATH = PLOTS_DIR / "precision_recall_curve.png"
LEARNING_RATE_CURVE_PATH = PLOTS_DIR / "learning_rate_curve.png"
GENERALIZATION_GAP_CURVE_PATH = PLOTS_DIR / "generalization_gap_curve.png"
VAL_CONFUSION_MATRIX_PATH = PLOTS_DIR / "val_confusion_matrix.png"
TEST_CONFUSION_MATRIX_PATH = PLOTS_DIR / "test_confusion_matrix.png"
PER_CLASS_F1_CURVE_PATH = PLOTS_DIR / "per_class_f1_curve.png"

IMAGE_SIZE = 512
NUM_CLASSES = 3
BATCH_SIZE = 8
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
RANDOM_SEED = 42

CLASS_TO_LABEL = {
    0: "Good",
    1: "Usable",
    2: "Reject",
}

TRAINING_SETTINGS = {
    "model": "ConvNeXt-Tiny",
    "pretrained": "ImageNet pretrained weights",
    "image_size": "512x512",
    "input_channels": "3 RGB",
    "final_layer": "Linear(768 -> 3)",
    "activation": "No Softmax inside model during training",
    "loss_function": "CrossEntropyLoss",
    "imbalance_handling": "Class weights calculated from train labels only",
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "scheduler": "ReduceLROnPlateau",
    "scheduler_monitor": "Validation loss",
    "main_model_selection_metric": "Validation Macro F1-score",
    "amp_mixed_precision": "Enabled only when CUDA is available",
}


class EyeQDataset(Dataset):
    # Dataset بسيط يقرأ الصور من CSV بدون تعديل ملفات البيانات.
    def __init__(self, csv_path, image_dir, transform=None):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.transform = transform
        self.dataframe = self.read_csv(csv_path)

    def read_csv(self, csv_path):
        # قراءة CSV والتحقق من الأعمدة المهمة.
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file was not found: {csv_path}")

        dataframe = pd.read_csv(csv_path)

        required_columns = ["image_name", "quality_label"]
        for column in required_columns:
            if column not in dataframe.columns:
                raise ValueError(f"Required column '{column}' was not found in {csv_path}")

        if "preprocessing_status" in dataframe.columns:
            dataframe = dataframe[dataframe["preprocessing_status"] == "Success"].copy()

        dataframe["quality_label"] = dataframe["quality_label"].astype(int)
        dataframe = dataframe[dataframe["quality_label"].isin([0, 1, 2])].copy()
        dataframe = dataframe.reset_index(drop=True)

        return dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self.find_image_path(row)

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["quality_label"])
        return image, label

    def find_image_path(self, row):
        # استخدام preprocessed_image_path إن كان صالحا، وإلا البحث داخل مجلد التقسيم.
        if "preprocessed_image_path" in row.index and pd.notna(row["preprocessed_image_path"]):
            candidate_path = Path(str(row["preprocessed_image_path"]))
            if candidate_path.exists():
                return candidate_path

        image_name = Path(str(row["image_name"])).stem
        candidate_path = self.image_dir / f"{image_name}.jpeg"

        if candidate_path.exists():
            return candidate_path

        raise FileNotFoundError(f"Image file was not found for image_name: {image_name}")


def create_output_folders():
    # إنشاء مجلدات نتائج التدريب المطلوبة فقط.
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed):
    # تثبيت العشوائية قدر الإمكان لتسهيل إعادة التجربة.
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    # اختيار CUDA إذا كان متاحا وإلا استخدام CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    # إنشاء transforms للتدريب والتحقق والاختبار بدون إعادة resize.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    return train_transform, eval_transform


def create_dataloaders():
    # إنشاء DataLoaders من ملفات CSV للصور المجهزة مسبقا.
    train_transform, eval_transform = get_transforms()

    train_dataset = EyeQDataset(TRAIN_CSV_PATH, TRAIN_IMAGE_DIR, transform=train_transform)
    val_dataset = EyeQDataset(VAL_CSV_PATH, VAL_IMAGE_DIR, transform=eval_transform)
    test_dataset = EyeQDataset(TEST_CSV_PATH, TEST_IMAGE_DIR, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def calculate_class_weights(train_dataset, device):
    # حساب class weights من تسميات train فقط.
    labels = train_dataset.dataframe["quality_label"].astype(int).values
    classes = np.array([0, 1, 2])

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    class_weights_df = pd.DataFrame(
        {
            "class_id": classes,
            "class_name": [CLASS_TO_LABEL[class_id] for class_id in classes],
            "class_weight": class_weights,
        }
    )
    class_weights_df.to_csv(CLASS_WEIGHTS_PATH, index=False)

    return torch.tensor(class_weights, dtype=torch.float32).to(device), class_weights_df


def load_model(device):
    # تحميل ConvNeXt-Tiny بأوزان ImageNet وتعديل classifier إلى 3 فئات.
    try:
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
        print("Loaded ConvNeXt-Tiny with ImageNet pretrained weights")
    except Exception as error:
        model = convnext_tiny(weights=None)
        print(f"Warning: ImageNet weights could not be loaded. Using random weights. Reason: {error}")

    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, NUM_CLASSES)

    model = model.to(device)
    return model


def calculate_metrics(labels, predictions, loss_value):
    # حساب accuracy و precision و recall و F1 الكلية ولكل فئة.
    accuracy = accuracy_score(labels, predictions)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    return {
        "loss": loss_value,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "good_precision": per_class_precision[0],
        "usable_precision": per_class_precision[1],
        "reject_precision": per_class_precision[2],
        "good_recall": per_class_recall[0],
        "usable_recall": per_class_recall[1],
        "reject_recall": per_class_recall[2],
        "good_f1": per_class_f1[0],
        "usable_f1": per_class_f1[1],
        "reject_f1": per_class_f1[2],
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp):
    # تنفيذ epoch تدريب واحد.
    model.train()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        predictions = torch.argmax(outputs, dim=1)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_predictions.extend(predictions.detach().cpu().numpy().tolist())

    average_loss = total_loss / len(dataloader.dataset)
    return calculate_metrics(all_labels, all_predictions, average_loss)


def evaluate_model(model, dataloader, criterion, device, use_amp):
    # تقييم النموذج بدون gradients.
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            predictions = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_predictions.extend(predictions.detach().cpu().numpy().tolist())

    average_loss = total_loss / len(dataloader.dataset)
    metrics = calculate_metrics(all_labels, all_predictions, average_loss)

    return metrics, all_labels, all_predictions


def get_current_learning_rate(optimizer):
    # قراءة learning rate الحالي من optimizer.
    return optimizer.param_groups[0]["lr"]


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_macro_f1, best_val_loss):
    # حفظ checkpoint يحتوي على النموذج وحالة التدريب الأساسية.
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_loss": best_val_loss,
        "class_to_label": CLASS_TO_LABEL,
        "training_settings": TRAINING_SETTINGS,
    }

    torch.save(checkpoint, path)


def load_checkpoint(path, device):
    # تحميل checkpoint مع دعم إصدارات PyTorch المختلفة.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def get_epoch_status_note(
    epoch,
    train_metrics,
    val_metrics,
    previous_train_f1,
    previous_val_f1,
    previous_val_loss,
    val_f1_improved,
    val_loss_improved,
    learning_rate_reduced,
    overfitting_counter,
):
    # إنشاء ملاحظة بسيطة عن حالة التدريب في هذا epoch.
    notes = []

    if val_f1_improved:
        notes.append("Validation improving")

    if learning_rate_reduced:
        notes.append("Learning rate reduced")

    if previous_train_f1 is not None and previous_val_f1 is not None and previous_val_loss is not None:
        train_improved = train_metrics["macro_f1"] > previous_train_f1
        val_got_worse = val_metrics["macro_f1"] < previous_val_f1 or val_metrics["loss"] > previous_val_loss

        if train_improved and val_got_worse:
            overfitting_counter += 1
        else:
            overfitting_counter = 0

        if overfitting_counter >= 2:
            notes.append("Possible overfitting")

    if train_metrics["macro_f1"] < 0.45 and val_metrics["macro_f1"] < 0.45 and epoch >= 3:
        notes.append("Possible underfitting")

    if not notes:
        if val_f1_improved or val_loss_improved:
            notes.append("OK")
        else:
            notes.append("OK")

    return "; ".join(notes), overfitting_counter


def print_epoch_summary(epoch, epoch_time, train_metrics, val_metrics, learning_rate, val_f1_improved, val_loss_improved, early_stopping_counter, status_note):
    # طباعة ملخص واضح لكل epoch.
    print(f"\nEpoch {epoch}/{EPOCHS}")
    print("-" * 60)
    print(f"Train loss: {train_metrics['loss']:.4f} | Val loss: {val_metrics['loss']:.4f}")
    print(f"Train accuracy: {train_metrics['accuracy']:.4f} | Val accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Train macro precision: {train_metrics['macro_precision']:.4f} | Val macro precision: {val_metrics['macro_precision']:.4f}")
    print(f"Train macro recall: {train_metrics['macro_recall']:.4f} | Val macro recall: {val_metrics['macro_recall']:.4f}")
    print(f"Train macro F1: {train_metrics['macro_f1']:.4f} | Val macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Val per-class F1: Good={val_metrics['good_f1']:.4f}, Usable={val_metrics['usable_f1']:.4f}, Reject={val_metrics['reject_f1']:.4f}")
    print(f"Learning rate: {learning_rate:.8f}")
    print(f"Epoch time seconds: {epoch_time:.2f}")
    print(f"Validation macro F1 improved: {val_f1_improved}")
    print(f"Validation loss improved: {val_loss_improved}")
    print(f"Early stopping counter: {early_stopping_counter}")
    print(f"Status note: {status_note}")


def build_epoch_row(epoch, epoch_time, train_metrics, val_metrics, learning_rate, val_f1_improved, val_loss_improved, early_stopping_counter, status_note):
    # تجهيز صف واحد لملف epoch_metrics.csv.
    generalization_gap = train_metrics["macro_f1"] - val_metrics["macro_f1"]
    loss_gap = val_metrics["loss"] - train_metrics["loss"]

    return {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "val_loss": val_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy": val_metrics["accuracy"],
        "train_macro_precision": train_metrics["macro_precision"],
        "val_macro_precision": val_metrics["macro_precision"],
        "train_macro_recall": train_metrics["macro_recall"],
        "val_macro_recall": val_metrics["macro_recall"],
        "train_macro_f1": train_metrics["macro_f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_good_f1": val_metrics["good_f1"],
        "val_usable_f1": val_metrics["usable_f1"],
        "val_reject_f1": val_metrics["reject_f1"],
        "learning_rate": learning_rate,
        "epoch_time_seconds": epoch_time,
        "val_macro_f1_improved": val_f1_improved,
        "val_loss_improved": val_loss_improved,
        "early_stopping_counter": early_stopping_counter,
        "generalization_gap": generalization_gap,
        "loss_gap": loss_gap,
        "status_note": status_note,
    }


def save_epoch_metrics(epoch_rows):
    # حفظ مقاييس epochs في CSV بعد كل epoch.
    pd.DataFrame(epoch_rows).to_csv(EPOCH_METRICS_PATH, index=False)


def plot_line_chart(x_values, y_series, title, xlabel, ylabel, output_path):
    # رسم منحنى بسيط من سلسلة أو أكثر.
    plt.figure(figsize=(8, 5))

    for label, values in y_series.items():
        plt.plot(x_values, values, marker="o", label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_training_plots(epoch_df):
    # إنشاء كل رسومات التدريب المطلوبة.
    epochs = epoch_df["epoch"].tolist()

    plot_line_chart(
        epochs,
        {"train loss": epoch_df["train_loss"], "val loss": epoch_df["val_loss"]},
        "Loss Curve",
        "Epoch",
        "Loss",
        LOSS_CURVE_PATH,
    )

    plot_line_chart(
        epochs,
        {"train accuracy": epoch_df["train_accuracy"], "val accuracy": epoch_df["val_accuracy"]},
        "Accuracy Curve",
        "Epoch",
        "Accuracy",
        ACCURACY_CURVE_PATH,
    )

    plot_line_chart(
        epochs,
        {"train macro F1": epoch_df["train_macro_f1"], "val macro F1": epoch_df["val_macro_f1"]},
        "Macro F1 Curve",
        "Epoch",
        "Macro F1",
        MACRO_F1_CURVE_PATH,
    )

    plot_line_chart(
        epochs,
        {"val macro precision": epoch_df["val_macro_precision"], "val macro recall": epoch_df["val_macro_recall"]},
        "Validation Precision and Recall",
        "Epoch",
        "Score",
        PRECISION_RECALL_CURVE_PATH,
    )

    plot_line_chart(
        epochs,
        {"learning rate": epoch_df["learning_rate"]},
        "Learning Rate Curve",
        "Epoch",
        "Learning Rate",
        LEARNING_RATE_CURVE_PATH,
    )

    plot_line_chart(
        epochs,
        {"generalization gap": epoch_df["generalization_gap"]},
        "Generalization Gap Curve",
        "Epoch",
        "Train Macro F1 - Val Macro F1",
        GENERALIZATION_GAP_CURVE_PATH,
    )

    plot_line_chart(
        epochs,
        {
            "Good": epoch_df["val_good_f1"],
            "Usable": epoch_df["val_usable_f1"],
            "Reject": epoch_df["val_reject_f1"],
        },
        "Validation Per-Class F1 Curve",
        "Epoch",
        "F1 Score",
        PER_CLASS_F1_CURVE_PATH,
    )


def plot_confusion_matrix(labels, predictions, title, output_path):
    # رسم confusion matrix باستخدام matplotlib فقط.
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    class_names = [CLASS_TO_LABEL[index] for index in [0, 1, 2]]
    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            color = "white" if matrix[row_index, column_index] > threshold else "black"
            plt.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color=color,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_classification_report_text(labels, predictions, output_path):
    # حفظ classification report كنص.
    report_text = classification_report(
        labels,
        predictions,
        labels=[0, 1, 2],
        target_names=[CLASS_TO_LABEL[0], CLASS_TO_LABEL[1], CLASS_TO_LABEL[2]],
        zero_division=0,
    )
    output_path.write_text(report_text, encoding="utf-8")


def save_test_metrics_summary(test_metrics):
    # حفظ ملخص مقاييس test في CSV.
    rows = [
        {"metric": "test_loss", "value": test_metrics["loss"]},
        {"metric": "test_accuracy", "value": test_metrics["accuracy"]},
        {"metric": "test_macro_precision", "value": test_metrics["macro_precision"]},
        {"metric": "test_macro_recall", "value": test_metrics["macro_recall"]},
        {"metric": "test_macro_f1", "value": test_metrics["macro_f1"]},
        {"metric": "good_precision", "value": test_metrics["good_precision"]},
        {"metric": "usable_precision", "value": test_metrics["usable_precision"]},
        {"metric": "reject_precision", "value": test_metrics["reject_precision"]},
        {"metric": "good_recall", "value": test_metrics["good_recall"]},
        {"metric": "usable_recall", "value": test_metrics["usable_recall"]},
        {"metric": "reject_recall", "value": test_metrics["reject_recall"]},
        {"metric": "good_f1", "value": test_metrics["good_f1"]},
        {"metric": "usable_f1", "value": test_metrics["usable_f1"]},
        {"metric": "reject_f1", "value": test_metrics["reject_f1"]},
    ]

    pd.DataFrame(rows).to_csv(TEST_METRICS_SUMMARY_PATH, index=False)


def get_class_distribution(dataset):
    # حساب توزيع الفئات داخل Dataset.
    counts = dataset.dataframe["quality_label"].value_counts().sort_index()
    lines = []

    for class_id in [0, 1, 2]:
        lines.append(f"{class_id} ({CLASS_TO_LABEL[class_id]}): {counts.get(class_id, 0)}")

    return "\n".join(lines)


def get_final_overfitting_analysis(epoch_df, best_epoch_by_f1):
    # تحليل نهائي بسيط لحالة overfitting أو underfitting.
    last_row = epoch_df.iloc[-1]
    best_row = epoch_df[epoch_df["epoch"] == best_epoch_by_f1].iloc[0]

    last_train_f1 = float(last_row["train_macro_f1"])
    last_val_f1 = float(last_row["val_macro_f1"])
    last_val_loss = float(last_row["val_loss"])
    last_train_loss = float(last_row["train_loss"])
    last_gap = last_train_f1 - last_val_f1
    last_loss_gap = last_val_loss - last_train_loss

    best_train_f1 = float(best_row["train_macro_f1"])
    best_val_f1 = float(best_row["val_macro_f1"])
    best_gap = best_train_f1 - best_val_f1

    if last_gap > 0.15 and last_loss_gap > 0.15:
        status = "Possible overfitting"
    elif last_train_f1 < 0.50 and last_val_f1 < 0.50:
        status = "Possible underfitting"
    elif best_val_f1 >= last_val_f1 and abs(best_gap) <= 0.15:
        status = "Training looks healthy"
    else:
        status = "Mixed behavior, review validation curves"

    lines = [
        status,
        f"Last epoch train macro F1: {last_train_f1:.4f}",
        f"Last epoch val macro F1: {last_val_f1:.4f}",
        f"Last epoch generalization gap: {last_gap:.4f}",
        f"Last epoch train loss: {last_train_loss:.4f}",
        f"Last epoch val loss: {last_val_loss:.4f}",
        f"Last epoch loss gap: {last_loss_gap:.4f}",
        f"Best epoch by macro F1: {best_epoch_by_f1}",
        f"Best epoch train macro F1: {best_train_f1:.4f}",
        f"Best epoch val macro F1: {best_val_f1:.4f}",
        f"Best epoch generalization gap: {best_gap:.4f}",
    ]

    return "\n".join(lines)


def get_created_files_list():
    # قائمة الملفات الناتجة المطلوبة.
    return [
        BEST_F1_MODEL_PATH,
        BEST_LOSS_MODEL_PATH,
        LAST_MODEL_PATH,
        EPOCH_METRICS_PATH,
        FINAL_REPORT_PATH,
        VAL_CLASSIFICATION_REPORT_PATH,
        TEST_CLASSIFICATION_REPORT_PATH,
        TEST_METRICS_SUMMARY_PATH,
        CLASS_WEIGHTS_PATH,
        LOSS_CURVE_PATH,
        ACCURACY_CURVE_PATH,
        MACRO_F1_CURVE_PATH,
        PRECISION_RECALL_CURVE_PATH,
        LEARNING_RATE_CURVE_PATH,
        GENERALIZATION_GAP_CURVE_PATH,
        VAL_CONFUSION_MATRIX_PATH,
        TEST_CONFUSION_MATRIX_PATH,
        PER_CLASS_F1_CURVE_PATH,
    ]


def write_final_training_report(
    train_dataset,
    val_dataset,
    test_dataset,
    class_weights_df,
    epoch_df,
    actual_epochs_completed,
    early_stopping_happened,
    best_val_macro_f1,
    best_val_loss,
    best_epoch_by_f1,
    best_epoch_by_loss,
    test_metrics,
):
    # كتابة التقرير النهائي للتدريب.
    final_analysis = get_final_overfitting_analysis(epoch_df, best_epoch_by_f1)

    lines = [
        "EyeQ ConvNeXt-Tiny 512x512 Final Training Report",
        "=" * 60,
        "",
        "Dataset paths:",
        f"Dataset root: {DATASET_ROOT}",
        f"Train folder: {TRAIN_IMAGE_DIR}",
        f"Val folder: {VAL_IMAGE_DIR}",
        f"Test folder: {TEST_IMAGE_DIR}",
        f"Train CSV: {TRAIN_CSV_PATH}",
        f"Val CSV: {VAL_CSV_PATH}",
        f"Test CSV: {TEST_CSV_PATH}",
        "",
        f"Number of train images: {len(train_dataset)}",
        f"Number of val images: {len(val_dataset)}",
        f"Number of test images: {len(test_dataset)}",
        "",
        "Class distribution in train:",
        get_class_distribution(train_dataset),
        "",
        "Class distribution in val:",
        get_class_distribution(val_dataset),
        "",
        "Class distribution in test:",
        get_class_distribution(test_dataset),
        "",
        "Class weights used:",
    ]

    for _, row in class_weights_df.iterrows():
        lines.append(f"{int(row['class_id'])} ({row['class_name']}): {row['class_weight']:.6f}")

    lines.extend(
        [
            "",
            f"Model name: {TRAINING_SETTINGS['model']}",
            f"Image size: {TRAINING_SETTINGS['image_size']}",
            f"Batch size: {BATCH_SIZE}",
            f"Epochs requested: {EPOCHS}",
            f"Actual epochs completed: {actual_epochs_completed}",
            f"Early stopping status: {'Happened' if early_stopping_happened else 'Did not happen'}",
            f"Best validation macro F1: {best_val_macro_f1:.6f}",
            f"Best validation loss: {best_val_loss:.6f}",
            f"Best epoch by macro F1: {best_epoch_by_f1}",
            f"Best epoch by val loss: {best_epoch_by_loss}",
            "",
            f"Final test accuracy: {test_metrics['accuracy']:.6f}",
            f"Final test macro precision: {test_metrics['macro_precision']:.6f}",
            f"Final test macro recall: {test_metrics['macro_recall']:.6f}",
            f"Final test macro F1: {test_metrics['macro_f1']:.6f}",
            "",
            "Final overfitting / underfitting analysis:",
            final_analysis,
            "",
            "Files created:",
        ]
    )

    for file_path in get_created_files_list():
        lines.append(str(file_path))

    lines.extend(
        [
            "",
            "Confirmation that no dataset files were modified: Yes",
        ]
    )

    FINAL_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, use_amp):
    # تشغيل التدريب مع early stopping وحفظ أفضل النماذج.
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch_by_f1 = 0
    best_epoch_by_loss = 0
    early_stopping_counter = 0
    early_stopping_happened = False
    epoch_rows = []
    overfitting_counter = 0

    previous_train_f1 = None
    previous_val_f1 = None
    previous_val_loss = None
    previous_learning_rate = get_current_learning_rate(optimizer)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device, use_amp)

        scheduler.step(val_metrics["loss"])

        current_learning_rate = get_current_learning_rate(optimizer)
        learning_rate_reduced = current_learning_rate < previous_learning_rate

        val_f1_improved = val_metrics["macro_f1"] > best_val_macro_f1
        val_loss_improved = val_metrics["loss"] < best_val_loss

        if val_f1_improved:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch_by_f1 = epoch
            save_checkpoint(BEST_F1_MODEL_PATH, epoch, model, optimizer, scheduler, best_val_macro_f1, best_val_loss)

        if val_loss_improved:
            best_val_loss = val_metrics["loss"]
            best_epoch_by_loss = epoch
            save_checkpoint(BEST_LOSS_MODEL_PATH, epoch, model, optimizer, scheduler, best_val_macro_f1, best_val_loss)

        if val_f1_improved:
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        epoch_time = time.time() - start_time

        status_note, overfitting_counter = get_epoch_status_note(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            previous_train_f1=previous_train_f1,
            previous_val_f1=previous_val_f1,
            previous_val_loss=previous_val_loss,
            val_f1_improved=val_f1_improved,
            val_loss_improved=val_loss_improved,
            learning_rate_reduced=learning_rate_reduced,
            overfitting_counter=overfitting_counter,
        )

        epoch_row = build_epoch_row(
            epoch=epoch,
            epoch_time=epoch_time,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=current_learning_rate,
            val_f1_improved=val_f1_improved,
            val_loss_improved=val_loss_improved,
            early_stopping_counter=early_stopping_counter,
            status_note=status_note,
        )
        epoch_rows.append(epoch_row)
        save_epoch_metrics(epoch_rows)

        print_epoch_summary(
            epoch=epoch,
            epoch_time=epoch_time,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=current_learning_rate,
            val_f1_improved=val_f1_improved,
            val_loss_improved=val_loss_improved,
            early_stopping_counter=early_stopping_counter,
            status_note=status_note,
        )

        previous_train_f1 = train_metrics["macro_f1"]
        previous_val_f1 = val_metrics["macro_f1"]
        previous_val_loss = val_metrics["loss"]
        previous_learning_rate = current_learning_rate

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            early_stopping_happened = True
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    save_checkpoint(LAST_MODEL_PATH, epoch_rows[-1]["epoch"], model, optimizer, scheduler, best_val_macro_f1, best_val_loss)

    return {
        "epoch_rows": epoch_rows,
        "actual_epochs_completed": epoch_rows[-1]["epoch"],
        "early_stopping_happened": early_stopping_happened,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_loss": best_val_loss,
        "best_epoch_by_f1": best_epoch_by_f1,
        "best_epoch_by_loss": best_epoch_by_loss,
    }


def evaluate_best_model(model, checkpoint_path, val_loader, test_loader, criterion, device, use_amp):
    # تحميل أفضل نموذج حسب Macro F1 وتقييمه على validation و test.
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    val_metrics, val_labels, val_predictions = evaluate_model(model, val_loader, criterion, device, use_amp)
    test_metrics, test_labels, test_predictions = evaluate_model(model, test_loader, criterion, device, use_amp)

    save_classification_report_text(val_labels, val_predictions, VAL_CLASSIFICATION_REPORT_PATH)
    save_classification_report_text(test_labels, test_predictions, TEST_CLASSIFICATION_REPORT_PATH)
    save_test_metrics_summary(test_metrics)

    plot_confusion_matrix(val_labels, val_predictions, "Validation Confusion Matrix", VAL_CONFUSION_MATRIX_PATH)
    plot_confusion_matrix(test_labels, test_predictions, "Test Confusion Matrix", TEST_CONFUSION_MATRIX_PATH)

    return val_metrics, test_metrics


def main():
    # نقطة تشغيل التدريب كاملة.
    print("Starting ConvNeXt-Tiny EyeQ 512x512 training")
    create_output_folders()
    set_random_seed(RANDOM_SEED)

    device = get_device()
    use_amp = device.type == "cuda"

    print(f"Device selected: {device}")
    print(f"AMP mixed precision enabled: {use_amp}")

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_dataloaders()

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    class_weights_tensor, class_weights_df = calculate_class_weights(train_dataset, device)
    print(f"Class weights saved: {CLASS_WEIGHTS_PATH}")

    model = load_model(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    training_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=use_amp,
    )

    epoch_df = pd.DataFrame(training_result["epoch_rows"])
    create_training_plots(epoch_df)

    print("\nEvaluating best model by validation Macro F1")
    _, test_metrics = evaluate_best_model(
        model=model,
        checkpoint_path=BEST_F1_MODEL_PATH,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )

    write_final_training_report(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        class_weights_df=class_weights_df,
        epoch_df=epoch_df,
        actual_epochs_completed=training_result["actual_epochs_completed"],
        early_stopping_happened=training_result["early_stopping_happened"],
        best_val_macro_f1=training_result["best_val_macro_f1"],
        best_val_loss=training_result["best_val_loss"],
        best_epoch_by_f1=training_result["best_epoch_by_f1"],
        best_epoch_by_loss=training_result["best_epoch_by_loss"],
        test_metrics=test_metrics,
    )

    print("\nFinal Summary")
    print("=" * 60)
    print("Training completed")
    print(f"Best validation macro F1: {training_result['best_val_macro_f1']:.6f}")
    print(f"Best validation loss: {training_result['best_val_loss']:.6f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.6f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.6f}")
    print(f"Early stopping happened: {training_result['early_stopping_happened']}")
    print(f"Reports folder path: {REPORTS_DIR}")
    print(f"Plots folder path: {PLOTS_DIR}")
    print(f"Best model path: {BEST_F1_MODEL_PATH}")


if __name__ == "__main__":
    main()