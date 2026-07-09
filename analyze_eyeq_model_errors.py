from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# هذا السكربت يحلل أخطاء نموذج ConvNeXt-Tiny على test set الخاص ببيانات EyeQ.
# يقوم بتحميل أفضل checkpoint، ثم يحسب التوقعات ودرجات الثقة واحتمالات الفئات.
# يحدد أنواع الأخطاء ويفصلها إلى أخطاء منخفضة ومتوسطة وعالية الخطورة.
# ينشئ تقارير CSV ورسومات وشبكات صور للأمثلة المهمة.
# لا يقوم هذا السكربت بتدريب النموذج ولا يعدل ملفات البيانات بأي شكل.


CHECKPOINT_PATH = Path(r"C:\EyePACS\training_runs\convnext_tiny_512_run1\checkpoints\best_model_by_macro_f1.pth")
TEST_CSV_PATH = Path(r"C:\EyePACS\EyeQ_Preprocessed_512\reports\test_preprocessed.csv")
TEST_IMAGES_DIR = Path(r"C:\EyePACS\EyeQ_Preprocessed_512\test")
TRAINING_RUN_DIR = Path(r"C:\EyePACS\training_runs\convnext_tiny_512_run1")

OUTPUT_DIR = TRAINING_RUN_DIR / "error_analysis"
PLOTS_DIR = OUTPUT_DIR / "plots"
SAMPLE_GRIDS_DIR = OUTPUT_DIR / "sample_grids"

PREDICTIONS_CSV_PATH = OUTPUT_DIR / "test_predictions_with_confidence.csv"
ERRORS_CSV_PATH = OUTPUT_DIR / "test_errors_only.csv"
HIGH_CONFIDENCE_ERRORS_CSV_PATH = OUTPUT_DIR / "high_confidence_errors.csv"
LOW_CONFIDENCE_PREDICTIONS_CSV_PATH = OUTPUT_DIR / "low_confidence_predictions.csv"
TOP_ERROR_EXAMPLES_CSV_PATH = OUTPUT_DIR / "top_error_examples.csv"
SUMMARY_REPORT_PATH = OUTPUT_DIR / "error_analysis_summary_report.txt"
FINAL_INTERPRETATION_PATH = OUTPUT_DIR / "final_error_interpretation.txt"

IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_CLASSES = 3
NUM_WORKERS = 0
HIGH_CONFIDENCE_THRESHOLD = 0.80
LOW_CONFIDENCE_THRESHOLD = 0.60

CLASS_TO_LABEL = {
    0: "Good",
    1: "Usable",
    2: "Reject",
}

ERROR_TYPES = [
    "Good_to_Usable",
    "Good_to_Reject",
    "Usable_to_Good",
    "Usable_to_Reject",
    "Reject_to_Good",
    "Reject_to_Usable",
]

GRID_ERROR_TYPES = [
    "Good_to_Usable",
    "Good_to_Reject",
    "Usable_to_Good",
    "Usable_to_Reject",
    "Reject_to_Good",
    "Reject_to_Usable",
]


class EyeQTestDataset(Dataset):
    # Dataset بسيط لقراءة صور test فقط بدون تعديل ملفات البيانات.
    def __init__(self, csv_path, image_folder, transform):
        self.csv_path = csv_path
        self.image_folder = image_folder
        self.transform = transform
        self.dataframe = self.read_test_csv(csv_path)

    def read_test_csv(self, csv_path):
        # قراءة CSV والتحقق من الأعمدة المهمة.
        if not csv_path.exists():
            raise FileNotFoundError(f"Critical error: test CSV was not found: {csv_path}")

        dataframe = pd.read_csv(csv_path)
        print(f"Loaded test CSV: {csv_path}")
        print(f"Rows found: {len(dataframe)}")
        print(f"Columns found: {list(dataframe.columns)}")

        required_columns = ["image_name", "quality_label"]
        for column in required_columns:
            if column not in dataframe.columns:
                raise ValueError(f"Critical error: required column '{column}' was not found in test CSV.")

        if "readable_quality_label" not in dataframe.columns:
            dataframe["readable_quality_label"] = dataframe["quality_label"].apply(
                lambda value: CLASS_TO_LABEL.get(int(value), "Unknown")
            )

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

        image_tensor = self.transform(image)
        true_label = int(row["quality_label"])
        image_name = Path(str(row["image_name"])).stem

        return image_tensor, true_label, image_name, str(image_path)

    def find_image_path(self, row):
        # استخدام preprocessed_image_path إذا كان صالحا، وإلا البحث داخل test folder بامتداد .jpeg.
        if "preprocessed_image_path" in row.index and pd.notna(row["preprocessed_image_path"]):
            candidate_path = Path(str(row["preprocessed_image_path"]))
            if candidate_path.exists() and candidate_path.is_file():
                return candidate_path

        image_name = Path(str(row["image_name"])).stem
        candidate_path = self.image_folder / f"{image_name}.jpeg"

        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path

        raise FileNotFoundError(f"Image was not found for image_name: {image_name}")


def create_output_folders():
    # إنشاء مجلدات الإخراج المطلوبة فقط.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_GRIDS_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    # اختيار CUDA إذا كانت متاحة وإلا استخدام CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_transform():
    # تحويلات test فقط بدون augmentation وبدون resize.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def create_test_loader():
    # إنشاء DataLoader للـ test set فقط.
    dataset = EyeQTestDataset(
        csv_path=TEST_CSV_PATH,
        image_folder=TEST_IMAGES_DIR,
        transform=get_test_transform(),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return dataset, dataloader


def load_checkpoint(path, device):
    # تحميل checkpoint مع دعم اختلاف إصدارات PyTorch.
    if not path.exists():
        raise FileNotFoundError(f"Critical error: checkpoint was not found: {path}")

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(device):
    # تحميل ConvNeXt-Tiny وتعديل classifier إلى 3 فئات ثم تحميل أوزان checkpoint.
    print("Loading ConvNeXt-Tiny model")

    model = convnext_tiny(weights=None)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(input_features, NUM_CLASSES)

    checkpoint = load_checkpoint(CHECKPOINT_PATH, device)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Critical error: checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    print("Model is in eval mode")

    return model


def get_error_type(true_label, predicted_label):
    # تحديد نوع التوقع سواء كان صحيحا أو خطأ.
    true_name = CLASS_TO_LABEL[true_label]
    predicted_name = CLASS_TO_LABEL[predicted_label]

    if true_label == predicted_label:
        return f"Correct_{true_name}"

    return f"{true_name}_to_{predicted_name}"


def get_clinical_risk_category(error_type):
    # تصنيف الأخطاء حسب الخطورة السريرية.
    if error_type.startswith("Correct_"):
        return "Correct"

    low_risk_errors = ["Good_to_Usable", "Reject_to_Usable"]
    medium_risk_errors = ["Usable_to_Good", "Usable_to_Reject", "Good_to_Reject"]
    high_risk_errors = ["Reject_to_Good"]

    if error_type in low_risk_errors:
        return "Low risk"

    if error_type in medium_risk_errors:
        return "Medium risk"

    if error_type in high_risk_errors:
        return "High risk"

    return "Unknown"


def run_inference(model, dataloader, device):
    # تشغيل inference على كل صور test وحساب الاحتمالات والثقة.
    print("Running inference on test set")

    rows = []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch_index, (images, labels, image_names, image_paths) in enumerate(dataloader, start=1):
            images = images.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)

            batch_confidences, batch_predictions = torch.max(probabilities, dim=1)

            probabilities_np = probabilities.detach().cpu().numpy()
            predictions_np = batch_predictions.detach().cpu().numpy()
            confidences_np = batch_confidences.detach().cpu().numpy()
            labels_np = labels.numpy()

            for index in range(len(labels_np)):
                true_label = int(labels_np[index])
                predicted_label = int(predictions_np[index])
                confidence = float(confidences_np[index])
                is_correct = true_label == predicted_label
                error_type = get_error_type(true_label, predicted_label)
                clinical_risk_category = get_clinical_risk_category(error_type)

                rows.append(
                    {
                        "image_name": image_names[index],
                        "image_path": image_paths[index],
                        "true_label": true_label,
                        "true_label_name": CLASS_TO_LABEL[true_label],
                        "predicted_label": predicted_label,
                        "predicted_label_name": CLASS_TO_LABEL[predicted_label],
                        "confidence": confidence,
                        "probability_good": float(probabilities_np[index][0]),
                        "probability_usable": float(probabilities_np[index][1]),
                        "probability_reject": float(probabilities_np[index][2]),
                        "is_correct": is_correct,
                        "error_type": error_type,
                        "clinical_risk_category": clinical_risk_category,
                    }
                )

            if batch_index % 20 == 0 or batch_index == len(dataloader):
                print(f"Processed batch {batch_index} / {len(dataloader)}")

    return pd.DataFrame(rows)


def save_prediction_csvs(predictions_df):
    # حفظ ملفات CSV الخاصة بالتوقعات والأخطاء.
    errors_df = predictions_df[predictions_df["is_correct"] == False].copy()
    high_confidence_errors_df = errors_df[errors_df["confidence"] >= HIGH_CONFIDENCE_THRESHOLD].copy()
    low_confidence_predictions_df = predictions_df[predictions_df["confidence"] < LOW_CONFIDENCE_THRESHOLD].copy()

    predictions_df.to_csv(PREDICTIONS_CSV_PATH, index=False)
    errors_df.to_csv(ERRORS_CSV_PATH, index=False)
    high_confidence_errors_df.to_csv(HIGH_CONFIDENCE_ERRORS_CSV_PATH, index=False)
    low_confidence_predictions_df.to_csv(LOW_CONFIDENCE_PREDICTIONS_CSV_PATH, index=False)

    top_high_confidence_wrong = errors_df.sort_values("confidence", ascending=False).head(30).copy()
    top_high_confidence_wrong["example_group"] = "highest_confidence_wrong"

    top_low_confidence_overall = predictions_df.sort_values("confidence", ascending=True).head(30).copy()
    top_low_confidence_overall["example_group"] = "lowest_confidence_overall"

    top_examples_df = pd.concat([top_high_confidence_wrong, top_low_confidence_overall], ignore_index=True)
    top_examples_df.to_csv(TOP_ERROR_EXAMPLES_CSV_PATH, index=False)

    return errors_df, high_confidence_errors_df, low_confidence_predictions_df, top_examples_df


def get_most_common_error_type(errors_df):
    # تحديد أكثر نوع خطأ شيوعا.
    if errors_df.empty:
        return "None"

    return errors_df["error_type"].value_counts().idxmax()


def calculate_accuracy(predictions_df):
    # حساب accuracy من التوقعات.
    return accuracy_score(predictions_df["true_label"], predictions_df["predicted_label"])


def format_error_distribution(errors_df):
    # تجهيز نص توزيع أنواع الأخطاء.
    if errors_df.empty:
        return "No wrong predictions."

    total_errors = len(errors_df)
    counts = errors_df["error_type"].value_counts()

    lines = []
    for error_type, count in counts.items():
        percentage = (count / total_errors) * 100 if total_errors else 0
        lines.append(f"{error_type}: {count} ({percentage:.2f}% of all errors)")

    return "\n".join(lines)


def format_average_confidence_by_column(predictions_df, column_name):
    # تجهيز نص متوسط الثقة حسب الفئة الحقيقية أو المتوقعة.
    grouped = predictions_df.groupby(column_name)["confidence"].mean()

    lines = []
    for label_id in [0, 1, 2]:
        value = grouped.get(label_id, np.nan)
        label_name = CLASS_TO_LABEL[label_id]

        if pd.isna(value):
            lines.append(f"{label_id} ({label_name}): Not available")
        else:
            lines.append(f"{label_id} ({label_name}): {value:.4f}")

    return "\n".join(lines)


def get_clinical_risk_summary(predictions_df):
    # حساب ملخص أخطاء الخطورة السريرية.
    counts = predictions_df["clinical_risk_category"].value_counts()

    return {
        "correct": int(counts.get("Correct", 0)),
        "low_risk": int(counts.get("Low risk", 0)),
        "medium_risk": int(counts.get("Medium risk", 0)),
        "high_risk": int(counts.get("High risk", 0)),
    }


def create_short_interpretation(predictions_df, errors_df, high_confidence_errors_df):
    # إنشاء تفسير مختصر مبني على الأرقام.
    if errors_df.empty:
        return "\n".join(
            [
                "No errors were found on the test set.",
                "The model did not show confusion between classes in this analysis.",
                "Reject_to_Good count is 0.",
                "There is no evidence of overconfident wrong predictions.",
            ]
        )

    most_common_error = get_most_common_error_type(errors_df)
    total_errors = len(errors_df)

    usable_related_errors = errors_df[
        errors_df["error_type"].isin(
            [
                "Good_to_Usable",
                "Usable_to_Good",
                "Usable_to_Reject",
                "Reject_to_Usable",
            ]
        )
    ]
    usable_confusion_ratio = len(usable_related_errors) / total_errors if total_errors else 0

    reject_to_good_count = int((errors_df["error_type"] == "Reject_to_Good").sum())
    reject_to_good_ratio = reject_to_good_count / total_errors if total_errors else 0

    correct_df = predictions_df[predictions_df["is_correct"] == True]
    wrong_df = predictions_df[predictions_df["is_correct"] == False]

    average_correct_confidence = correct_df["confidence"].mean() if not correct_df.empty else 0
    average_wrong_confidence = wrong_df["confidence"].mean() if not wrong_df.empty else 0

    if usable_confusion_ratio >= 0.50:
        usable_message = "The model is mostly confusing borderline Usable-related images."
    else:
        usable_message = "The model is not mostly limited to Usable-related confusion."

    if reject_to_good_count == 0:
        reject_message = "Reject_to_Good is absent in this test analysis."
    elif reject_to_good_ratio <= 0.10:
        reject_message = "Reject_to_Good is rare compared with all errors."
    else:
        reject_message = "Reject_to_Good is common enough to require careful review."

    if average_wrong_confidence >= 0.80 or len(high_confidence_errors_df) > 0:
        confidence_message = "The model shows overconfidence on some wrong predictions."
    elif average_wrong_confidence >= average_correct_confidence:
        confidence_message = "Wrong predictions have confidence comparable to correct predictions."
    else:
        confidence_message = "Wrong predictions are generally less confident than correct predictions."

    return "\n".join(
        [
            usable_message,
            reject_message,
            f"Most frequent error type: {most_common_error}",
            confidence_message,
            f"Average confidence for correct predictions: {average_correct_confidence:.4f}",
            f"Average confidence for wrong predictions: {average_wrong_confidence:.4f}",
        ]
    )


def write_summary_report(predictions_df, errors_df, high_confidence_errors_df, low_confidence_predictions_df, skipped_grids, created_files):
    # كتابة تقرير ملخص تحليل الأخطاء.
    total_images = len(predictions_df)
    correct_count = int(predictions_df["is_correct"].sum())
    wrong_count = len(errors_df)
    accuracy = calculate_accuracy(predictions_df) if total_images else 0
    error_rate = wrong_count / total_images if total_images else 0

    correct_df = predictions_df[predictions_df["is_correct"] == True]
    wrong_df = predictions_df[predictions_df["is_correct"] == False]

    average_correct_confidence = correct_df["confidence"].mean() if not correct_df.empty else 0
    average_wrong_confidence = wrong_df["confidence"].mean() if not wrong_df.empty else 0

    risk_summary = get_clinical_risk_summary(predictions_df)
    reject_to_good_count = int((errors_df["error_type"] == "Reject_to_Good").sum())

    lines = [
        "EyeQ ConvNeXt-Tiny Error Analysis Summary Report",
        "=" * 60,
        f"Total test images: {total_images}",
        f"Correct predictions count: {correct_count}",
        f"Wrong predictions count: {wrong_count}",
        f"Accuracy: {accuracy:.6f}",
        f"Error rate: {error_rate:.6f}",
        "",
        "Number of errors per error type and percentage from all errors:",
        format_error_distribution(errors_df),
        "",
        f"Number of high-confidence errors: {len(high_confidence_errors_df)}",
        f"Number of low-confidence predictions: {len(low_confidence_predictions_df)}",
        f"Average confidence for correct predictions: {average_correct_confidence:.6f}",
        f"Average confidence for wrong predictions: {average_wrong_confidence:.6f}",
        "",
        "Average confidence per true class:",
        format_average_confidence_by_column(predictions_df, "true_label"),
        "",
        "Average confidence per predicted class:",
        format_average_confidence_by_column(predictions_df, "predicted_label"),
        "",
        "Clinical risk summary:",
        f"Correct count: {risk_summary['correct']}",
        f"Low risk error count: {risk_summary['low_risk']}",
        f"Medium risk error count: {risk_summary['medium_risk']}",
        f"High risk error count: {risk_summary['high_risk']}",
        "",
        "Specific high-risk count:",
        f"Reject_to_Good count: {reject_to_good_count}",
        "",
        "Short interpretation:",
        create_short_interpretation(predictions_df, errors_df, high_confidence_errors_df),
        "",
        "Skipped sample grids because no matching images existed:",
    ]

    if skipped_grids:
        lines.extend(skipped_grids)
    else:
        lines.append("None")

    lines.extend(
        [
            "",
            "Created files:",
        ]
    )

    for file_path in created_files:
        lines.append(str(file_path))

    lines.extend(
        [
            "",
            "Confirmation that no training was performed: Yes",
            "Confirmation that dataset files were not modified: Yes",
        ]
    )

    SUMMARY_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_final_interpretation(predictions_df, errors_df, high_confidence_errors_df, created_files):
    # كتابة تقرير تفسير نهائي بلغة إنجليزية واضحة.
    if errors_df.empty:
        most_common_error = "None"
        reject_to_good_count = 0
        usable_is_confusing = False
    else:
        most_common_error = get_most_common_error_type(errors_df)
        reject_to_good_count = int((errors_df["error_type"] == "Reject_to_Good").sum())

        usable_related_errors = errors_df[
            errors_df["error_type"].isin(
                [
                    "Good_to_Usable",
                    "Usable_to_Good",
                    "Usable_to_Reject",
                    "Reject_to_Usable",
                ]
            )
        ]
        usable_is_confusing = len(usable_related_errors) >= (len(errors_df) / 2)

    good_reject_errors = int(
        errors_df["error_type"].isin(["Good_to_Reject", "Reject_to_Good"]).sum()
    ) if not errors_df.empty else 0

    if errors_df.empty:
        main_pattern = "The model made no mistakes on the analyzed test set."
    elif usable_is_confusing:
        main_pattern = "The main error pattern involves Usable-related borderline confusion."
    else:
        main_pattern = "The errors are spread across multiple class transitions."

    if good_reject_errors == 0:
        safety_message = "The model safely separates Good and Reject in this test analysis."
    elif reject_to_good_count == 0:
        safety_message = "The model has some Good/Reject confusion, but no Reject images were predicted as Good."
    else:
        safety_message = "The model does not perfectly separate Good and Reject because some Reject images were predicted as Good."

    if not errors_df.empty and len(high_confidence_errors_df) > 0:
        confidence_message = "Some wrong predictions are high-confidence, so these examples should be reviewed carefully."
    else:
        confidence_message = "High-confidence wrong predictions were not prominent in this analysis."

    lines = [
        "Final EyeQ Error Interpretation",
        "=" * 60,
        "",
        f"Main error pattern: {main_pattern}",
        f"Most common error type: {most_common_error}",
        f"Whether Usable is the most confusing class: {'Yes' if usable_is_confusing else 'No'}",
        f"Whether the model safely separates Good and Reject: {safety_message}",
        f"How many Reject images were predicted as Good: {reject_to_good_count}",
        "",
        "Why Usable is expected to be hard:",
        "Usable images are often borderline cases between clearly acceptable Good images and clearly unacceptable Reject images.",
        "This middle category can contain mild blur, partial artifacts, uneven illumination, or other quality issues that are not always visually extreme.",
        "",
        "What this means for a retinal image quality assessment system:",
        "The system should be reviewed carefully on borderline Usable samples because these are likely to drive many practical disagreements.",
        "High-risk errors such as Reject_to_Good matter most because they may allow poor-quality retinal images to pass downstream analysis.",
        confidence_message,
        "",
        "Suggested next improvements:",
        "1. Review borderline Usable samples.",
        "2. Add more examples for confusing classes if available.",
        "3. Consider binary Acceptable vs Reject as a secondary experiment.",
        "4. Consider threshold-based rejection if high-risk errors matter.",
        "5. Consider Grad-CAM in a later explainability stage.",
        "",
        "Created files:",
    ]

    for file_path in created_files:
        lines.append(str(file_path))

    FINAL_INTERPRETATION_PATH.write_text("\n".join(lines), encoding="utf-8")


def save_bar_plot(labels, values, title, xlabel, ylabel, output_path, rotation=0):
    # حفظ رسم أعمدة بسيط.
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values, color="#1565C0")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_error_type_distribution_plot(errors_df):
    # رسم توزيع أنواع الأخطاء.
    output_path = PLOTS_DIR / "error_type_distribution.png"

    if errors_df.empty:
        labels = ["No errors"]
        values = [0]
    else:
        counts = errors_df["error_type"].value_counts()
        labels = counts.index.tolist()
        values = counts.values.tolist()

    save_bar_plot(
        labels=labels,
        values=values,
        title="Error Type Distribution",
        xlabel="Error type",
        ylabel="Count",
        output_path=output_path,
        rotation=45,
    )

    return output_path


def create_clinical_risk_distribution_plot(predictions_df):
    # رسم توزيع فئات الخطورة السريرية.
    output_path = PLOTS_DIR / "clinical_risk_distribution.png"
    risk_order = ["Correct", "Low risk", "Medium risk", "High risk"]
    counts = predictions_df["clinical_risk_category"].value_counts()
    values = [counts.get(risk, 0) for risk in risk_order]

    save_bar_plot(
        labels=risk_order,
        values=values,
        title="Clinical Risk Distribution",
        xlabel="Clinical risk category",
        ylabel="Count",
        output_path=output_path,
        rotation=0,
    )

    return output_path


def create_confidence_correct_vs_wrong_plot(predictions_df):
    # مقارنة توزيع الثقة بين التوقعات الصحيحة والخاطئة.
    output_path = PLOTS_DIR / "confidence_correct_vs_wrong.png"

    correct_confidence = predictions_df[predictions_df["is_correct"] == True]["confidence"].values
    wrong_confidence = predictions_df[predictions_df["is_correct"] == False]["confidence"].values

    plt.figure(figsize=(8, 5))
    plt.hist(correct_confidence, bins=20, alpha=0.7, label="Correct", color="#2E7D32")
    plt.hist(wrong_confidence, bins=20, alpha=0.7, label="Wrong", color="#C62828")
    plt.title("Confidence Distribution: Correct vs Wrong")
    plt.xlabel("Confidence")
    plt.ylabel("Image count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def create_confidence_by_true_class_plot(predictions_df):
    # رسم متوسط الثقة حسب الفئة الحقيقية.
    output_path = PLOTS_DIR / "confidence_by_true_class.png"
    grouped = predictions_df.groupby("true_label")["confidence"].mean()

    labels = [CLASS_TO_LABEL[label_id] for label_id in [0, 1, 2]]
    values = [grouped.get(label_id, 0) for label_id in [0, 1, 2]]

    save_bar_plot(
        labels=labels,
        values=values,
        title="Average Confidence by True Class",
        xlabel="True class",
        ylabel="Average confidence",
        output_path=output_path,
        rotation=0,
    )

    return output_path


def create_confidence_by_predicted_class_plot(predictions_df):
    # رسم متوسط الثقة حسب الفئة المتوقعة.
    output_path = PLOTS_DIR / "confidence_by_predicted_class.png"
    grouped = predictions_df.groupby("predicted_label")["confidence"].mean()

    labels = [CLASS_TO_LABEL[label_id] for label_id in [0, 1, 2]]
    values = [grouped.get(label_id, 0) for label_id in [0, 1, 2]]

    save_bar_plot(
        labels=labels,
        values=values,
        title="Average Confidence by Predicted Class",
        xlabel="Predicted class",
        ylabel="Average confidence",
        output_path=output_path,
        rotation=0,
    )

    return output_path


def create_probability_boxplot_by_true_class(predictions_df):
    # رسم boxplot للثقة حسب الفئة الحقيقية.
    output_path = PLOTS_DIR / "probability_boxplot_by_true_class.png"

    data = []
    labels = []

    for label_id in [0, 1, 2]:
        values = predictions_df[predictions_df["true_label"] == label_id]["confidence"].values
        if len(values) > 0:
            data.append(values)
            labels.append(CLASS_TO_LABEL[label_id])

    plt.figure(figsize=(8, 5))

    if data:
        plt.boxplot(data, tick_labels=labels)
    else:
        plt.boxplot([[0]], tick_labels=["No data"])

    plt.title("Confidence Boxplot by True Class")
    plt.xlabel("True class")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def create_high_confidence_errors_by_type_plot(high_confidence_errors_df):
    # رسم الأخطاء عالية الثقة حسب نوع الخطأ.
    output_path = PLOTS_DIR / "high_confidence_errors_by_type.png"

    if high_confidence_errors_df.empty:
        labels = ["No high-confidence errors"]
        values = [0]
    else:
        counts = high_confidence_errors_df["error_type"].value_counts()
        labels = counts.index.tolist()
        values = counts.values.tolist()

    save_bar_plot(
        labels=labels,
        values=values,
        title="High-Confidence Errors by Type",
        xlabel="Error type",
        ylabel="Count",
        output_path=output_path,
        rotation=45,
    )

    return output_path


def create_all_plots(predictions_df, errors_df, high_confidence_errors_df):
    # إنشاء كل الرسومات المطلوبة.
    created_plot_paths = [
        create_error_type_distribution_plot(errors_df),
        create_clinical_risk_distribution_plot(predictions_df),
        create_confidence_correct_vs_wrong_plot(predictions_df),
        create_confidence_by_true_class_plot(predictions_df),
        create_confidence_by_predicted_class_plot(predictions_df),
        create_probability_boxplot_by_true_class(predictions_df),
        create_high_confidence_errors_by_type_plot(high_confidence_errors_df),
    ]

    return created_plot_paths


def create_sample_grid(dataframe, output_path, title):
    # إنشاء شبكة صور حتى 12 صورة للعينات المختارة بدون تعديل الصور.
    if dataframe.empty:
        return False

    sample_df = dataframe.head(12).copy()
    image_count = len(sample_df)
    columns = 3
    rows = int(np.ceil(image_count / columns))

    plt.figure(figsize=(columns * 4, rows * 4))

    for index, (_, row) in enumerate(sample_df.iterrows(), start=1):
        image_path = Path(row["image_path"])

        if not image_path.exists():
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        axis = plt.subplot(rows, columns, index)
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"True: {row['true_label_name']}\nPred: {row['predicted_label_name']}\nConf: {row['confidence']:.2f}",
            fontsize=10,
        )

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return True


def create_sample_grids(errors_df, high_confidence_errors_df, low_confidence_predictions_df):
    # إنشاء شبكات الصور للأخطاء المهمة إذا وجدت.
    created_grid_paths = []
    skipped_grids = []

    for error_type in GRID_ERROR_TYPES:
        output_path = SAMPLE_GRIDS_DIR / f"{error_type}_examples.png"
        subset_df = errors_df[errors_df["error_type"] == error_type].sort_values("confidence", ascending=False)

        created = create_sample_grid(
            dataframe=subset_df,
            output_path=output_path,
            title=f"{error_type} Examples",
        )

        if created:
            created_grid_paths.append(output_path)
        else:
            skipped_grids.append(f"{error_type}_examples.png")

    high_confidence_output_path = SAMPLE_GRIDS_DIR / "high_confidence_errors_examples.png"
    high_confidence_subset = high_confidence_errors_df.sort_values("confidence", ascending=False)

    if create_sample_grid(high_confidence_subset, high_confidence_output_path, "High-Confidence Error Examples"):
        created_grid_paths.append(high_confidence_output_path)
    else:
        skipped_grids.append("high_confidence_errors_examples.png")

    low_confidence_output_path = SAMPLE_GRIDS_DIR / "low_confidence_predictions_examples.png"
    low_confidence_subset = low_confidence_predictions_df.sort_values("confidence", ascending=True)

    if create_sample_grid(low_confidence_subset, low_confidence_output_path, "Low-Confidence Prediction Examples"):
        created_grid_paths.append(low_confidence_output_path)
    else:
        skipped_grids.append("low_confidence_predictions_examples.png")

    return created_grid_paths, skipped_grids


def get_created_files(plot_paths, grid_paths):
    # جمع قائمة الملفات التي تم إنشاؤها فعليا.
    created_files = [
        PREDICTIONS_CSV_PATH,
        ERRORS_CSV_PATH,
        HIGH_CONFIDENCE_ERRORS_CSV_PATH,
        LOW_CONFIDENCE_PREDICTIONS_CSV_PATH,
        TOP_ERROR_EXAMPLES_CSV_PATH,
        SUMMARY_REPORT_PATH,
        FINAL_INTERPRETATION_PATH,
    ]

    created_files.extend(plot_paths)
    created_files.extend(grid_paths)

    return created_files


def print_final_summary(predictions_df, errors_df):
    # طباعة الملخص النهائي المطلوب.
    total_test_images = len(predictions_df)
    correct_predictions = int(predictions_df["is_correct"].sum())
    wrong_predictions = len(errors_df)
    accuracy = calculate_accuracy(predictions_df) if total_test_images else 0
    most_common_error_type = get_most_common_error_type(errors_df)
    high_confidence_error_count = int(
        ((predictions_df["is_correct"] == False) & (predictions_df["confidence"] >= HIGH_CONFIDENCE_THRESHOLD)).sum()
    )
    reject_to_good_count = int((errors_df["error_type"] == "Reject_to_Good").sum())

    print("\nFinal Summary")
    print("=" * 60)
    print(f"Total test images: {total_test_images}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Wrong predictions: {wrong_predictions}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Most common error type: {most_common_error_type}")
    print(f"High-confidence error count: {high_confidence_error_count}")
    print(f"Reject_to_Good count: {reject_to_good_count}")
    print(f"Error analysis folder path: {OUTPUT_DIR}")
    print("Confirmation that no training was performed: Yes")
    print("Confirmation that dataset files were not modified: Yes")


def main():
    # تشغيل تحليل الأخطاء كاملا بدون تدريب وبدون تعديل البيانات.
    print("Starting EyeQ model error analysis")
    create_output_folders()

    device = get_device()
    print(f"Device selected: {device}")
    print(f"AMP enabled: {device.type == 'cuda'}")

    test_dataset, test_loader = create_test_loader()
    print(f"Total test images to analyze: {len(test_dataset)}")

    model = load_model(device)

    predictions_df = run_inference(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    errors_df, high_confidence_errors_df, low_confidence_predictions_df, _ = save_prediction_csvs(predictions_df)

    print("Creating plots")
    plot_paths = create_all_plots(
        predictions_df=predictions_df,
        errors_df=errors_df,
        high_confidence_errors_df=high_confidence_errors_df,
    )

    print("Creating image sample grids")
    grid_paths, skipped_grids = create_sample_grids(
        errors_df=errors_df,
        high_confidence_errors_df=high_confidence_errors_df,
        low_confidence_predictions_df=low_confidence_predictions_df,
    )

    created_files = get_created_files(plot_paths, grid_paths)

    print("Writing reports")
    write_summary_report(
        predictions_df=predictions_df,
        errors_df=errors_df,
        high_confidence_errors_df=high_confidence_errors_df,
        low_confidence_predictions_df=low_confidence_predictions_df,
        skipped_grids=skipped_grids,
        created_files=created_files,
    )

    write_final_interpretation(
        predictions_df=predictions_df,
        errors_df=errors_df,
        high_confidence_errors_df=high_confidence_errors_df,
        created_files=created_files,
    )

    print("Created files:")
    for file_path in created_files:
        print(file_path)

    print_final_summary(predictions_df, errors_df)


if __name__ == "__main__":
    main()
