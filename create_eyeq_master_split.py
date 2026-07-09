from pathlib import Path
import shutil
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# هذا السكربت يدمج تسميات جودة EyeQ من ملفي Label_EyeQ_train.csv و Label_EyeQ_test.csv.
# يستخدم الصور الموجودة فقط داخل Final Train و Final Test، ولا يستخدم مجلدات train و test الأصلية.
# ينشئ ملف manifest شامل، ثم يقسم الصور إلى train و val و test مع منع تسرب نفس المريض بين التقسيمات.
# المخرجات تكون داخل EyeQ_Split_Dataset، وتشمل الصور المنسوخة وملفات CSV والتقرير والرسومات.
# المكتبات المستخدمة: pathlib للمسارات، pandas لملفات CSV، shutil لنسخ الصور، matplotlib للرسم، sklearn للتقسيم، collections للعد.


BASE_DIR = Path(r"C:\EyePACS")

FINAL_TRAIN_DIR = BASE_DIR / "Final Train"
FINAL_TEST_DIR = BASE_DIR / "Final Test"

TRAIN_CSV_PATH = BASE_DIR / "Label_EyeQ_train.csv"
TEST_CSV_PATH = BASE_DIR / "Label_EyeQ_test.csv"

OUTPUT_DIR = BASE_DIR / "EyeQ_Split_Dataset"
OUTPUT_TRAIN_DIR = OUTPUT_DIR / "train"
OUTPUT_VAL_DIR = OUTPUT_DIR / "val"
OUTPUT_TEST_DIR = OUTPUT_DIR / "test"
REPORTS_DIR = OUTPUT_DIR / "reports"

MASTER_MANIFEST_PATH = REPORTS_DIR / "master_eyeq_quality_manifest.csv"
TRAIN_SPLIT_CSV_PATH = REPORTS_DIR / "train_split.csv"
VAL_SPLIT_CSV_PATH = REPORTS_DIR / "val_split.csv"
TEST_SPLIT_CSV_PATH = REPORTS_DIR / "test_split.csv"
SUMMARY_REPORT_PATH = REPORTS_DIR / "split_summary_report.txt"

OVERALL_CLASS_PLOT_PATH = REPORTS_DIR / "overall_class_distribution.png"
SPLIT_CLASS_PLOT_PATH = REPORTS_DIR / "split_class_distribution.png"
SPLIT_PERCENTAGES_PLOT_PATH = REPORTS_DIR / "split_percentages.png"

SUPPORTED_EXTENSIONS = [".jpeg", ".jpg", ".png"]

QUALITY_LABEL_MAP = {
    0: "Good",
    1: "Usable",
    2: "Reject",
}


def create_output_folders():
    # إنشاء مجلدات الإخراج المطلوبة فقط.
    OUTPUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_VAL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEST_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def detect_image_column(columns):
    # محاولة اكتشاف عمود اسم الصورة من أسماء الأعمدة الشائعة.
    preferred_names = [
        "image",
        "image_name",
        "filename",
        "file_name",
        "name",
        "img",
        "fundus",
    ]

    lower_columns = {column.lower().strip(): column for column in columns}

    for name in preferred_names:
        if name in lower_columns:
            return lower_columns[name]

    for column in columns:
        column_lower = column.lower().strip()
        if "image" in column_lower or "file" in column_lower or "name" in column_lower:
            return column

    return None


def detect_quality_column(columns, image_column):
    # محاولة اكتشاف عمود تسمية الجودة من أسماء الأعمدة الشائعة.
    preferred_names = [
        "quality",
        "quality_label",
        "label",
        "class",
        "grade",
        "eyeq",
        "image_quality",
    ]

    lower_columns = {column.lower().strip(): column for column in columns}

    for name in preferred_names:
        if name in lower_columns and lower_columns[name] != image_column:
            return lower_columns[name]

    for column in columns:
        if column == image_column:
            continue

        column_lower = column.lower().strip()
        if "quality" in column_lower or "label" in column_lower or "class" in column_lower:
            return column

    return None


def clean_image_name(value):
    # تنظيف اسم الصورة وإزالة أي مسار أو امتداد إن وجد.
    return Path(str(value).strip()).stem


def normalize_quality_label(value):
    # تحويل تسمية الجودة إلى رقم صحيح مع الإبلاغ عن القيم غير الصالحة لاحقا.
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def get_readable_quality_label(quality_label):
    # تحويل التسمية الرقمية إلى نص مفهوم.
    return QUALITY_LABEL_MAP.get(quality_label, "Unknown")


def extract_patient_and_eye(image_name):
    # استخراج patient_id و eye_side حسب قاعدة أسماء الصور مثل 10_left و 10_right.
    clean_name = clean_image_name(image_name)

    if clean_name.endswith("_left"):
        return clean_name[:-5], "left"

    if clean_name.endswith("_right"):
        return clean_name[:-6], "right"

    return clean_name, "unknown"


def find_image_in_final_folders(image_name):
    # البحث عن الصورة داخل Final Train و Final Test فقط وبالامتدادات المدعومة.
    image_stem = clean_image_name(image_name)

    for folder_name, folder_path in [("Final Train", FINAL_TRAIN_DIR), ("Final Test", FINAL_TEST_DIR)]:
        for extension in SUPPORTED_EXTENSIONS:
            candidate_path = folder_path / f"{image_stem}{extension}"
            if candidate_path.exists():
                return folder_name, candidate_path

    return "", ""


def read_label_csv(csv_path):
    # قراءة ملف CSV وفحص الأعمدة قبل استخدامها.
    if not csv_path.exists():
        raise FileNotFoundError(f"Critical error: required CSV file was not found: {csv_path}")

    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Columns in {csv_path.name}: {list(df.columns)}")

    image_column = detect_image_column(df.columns)
    if image_column is None:
        raise ValueError(f"Critical error: image-name column could not be detected in {csv_path.name}")

    quality_column = detect_quality_column(df.columns, image_column)
    if quality_column is None:
        raise ValueError(f"Critical error: quality-label column could not be detected in {csv_path.name}")

    print(f"Detected image column: {image_column}")
    print(f"Detected quality label column: {quality_column}")

    return df, image_column, quality_column


def build_manifest_from_csv(csv_path):
    # بناء manifest من ملف CSV واحد مع تسجيل الصور الموجودة والمفقودة.
    df, image_column, quality_column = read_label_csv(csv_path)

    manifest_rows = []

    for index, row in df.iterrows():
        image_name = clean_image_name(row[image_column])
        quality_label = normalize_quality_label(row[quality_column])
        readable_quality_label = get_readable_quality_label(quality_label)
        patient_id, eye_side = extract_patient_and_eye(image_name)
        source_folder, image_path = find_image_in_final_folders(image_name)

        found_status = "Found" if image_path else "Missing"

        manifest_rows.append(
            {
                "image_name": image_name,
                "image_path": str(image_path) if image_path else "",
                "original_source_csv": csv_path.name,
                "original_source_folder": source_folder,
                "quality_label": quality_label,
                "readable_quality_label": readable_quality_label,
                "patient_id": patient_id,
                "eye_side": eye_side,
                "found_status": found_status,
            }
        )

        if (index + 1) % 500 == 0 or (index + 1) == len(df):
            print(f"{csv_path.name}: processed {index + 1} / {len(df)} rows")

    manifest_df = pd.DataFrame(manifest_rows)
    return df, manifest_df


def validate_quality_labels(found_df):
    # التأكد من أن التسميات الموجودة هي 0 و 1 و 2 فقط.
    labels = set(found_df["quality_label"].dropna().unique())
    invalid_labels = sorted(labels - set(QUALITY_LABEL_MAP.keys()))

    if invalid_labels:
        raise ValueError(f"Critical error: unexpected quality labels found: {invalid_labels}")


def representative_label(labels):
    # اختيار أكثر تسمية جودة تكرارا للمريض لاستخدامها في التقسيم الطبقي.
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def can_use_stratify(labels):
    # تحديد هل يمكن استخدام stratify بدون فشل بسبب قلة العينات.
    counts = Counter(labels)
    return len(counts) > 1 and min(counts.values()) >= 2


def safe_train_test_split(dataframe, test_size, stratify_labels):
    # تنفيذ تقسيم طبقي إن أمكن، ثم الرجوع لتقسيم عادي إذا كان التقسيم الطبقي غير ممكن.
    try:
        return train_test_split(
            dataframe,
            test_size=test_size,
            random_state=42,
            shuffle=True,
            stratify=stratify_labels,
        )
    except ValueError as error:
        print(f"Warning: stratified split was not possible. Using non-stratified split. Reason: {error}")
        return train_test_split(
            dataframe,
            test_size=test_size,
            random_state=42,
            shuffle=True,
            stratify=None,
        )


def split_patients(found_df):
    # تقسيم المرضى إلى train و val و test مع منع وجود نفس المريض في أكثر من تقسيم.
    patient_rows = []

    for patient_id, group in found_df.groupby("patient_id"):
        patient_rows.append(
            {
                "patient_id": patient_id,
                "representative_label": representative_label(group["quality_label"].tolist()),
            }
        )

    patient_df = pd.DataFrame(patient_rows)

    if len(patient_df) < 3:
        raise ValueError("Critical error: not enough unique patients to create train / val / test splits.")

    first_stratify = (
        patient_df["representative_label"]
        if can_use_stratify(patient_df["representative_label"])
        else None
    )

    train_patients, temp_patients = safe_train_test_split(
        dataframe=patient_df,
        test_size=0.30,
        stratify_labels=first_stratify,
    )

    second_stratify = (
        temp_patients["representative_label"]
        if can_use_stratify(temp_patients["representative_label"])
        else None
    )

    val_patients, test_patients = safe_train_test_split(
        dataframe=temp_patients,
        test_size=0.50,
        stratify_labels=second_stratify,
    )

    train_ids = set(train_patients["patient_id"])
    val_ids = set(val_patients["patient_id"])
    test_ids = set(test_patients["patient_id"])

    train_split = found_df[found_df["patient_id"].isin(train_ids)].copy()
    val_split = found_df[found_df["patient_id"].isin(val_ids)].copy()
    test_split = found_df[found_df["patient_id"].isin(test_ids)].copy()

    train_split["new_split"] = "train"
    val_split["new_split"] = "val"
    test_split["new_split"] = "test"

    return train_split, val_split, test_split


def check_patient_leakage(train_split, val_split, test_split):
    # التحقق من عدم تكرار patient_id بين التقسيمات.
    train_ids = set(train_split["patient_id"])
    val_ids = set(val_split["patient_id"])
    test_ids = set(test_split["patient_id"])

    has_leakage = bool(train_ids & val_ids) or bool(train_ids & test_ids) or bool(val_ids & test_ids)
    return not has_leakage


def copy_split_images(split_df, output_folder):
    # نسخ الصور إلى مجلد التقسيم الجديد بدون تغيير أسماء الصور.
    copied_paths = []

    for _, row in split_df.iterrows():
        source_path = Path(row["image_path"])

        if not source_path.exists():
            raise FileNotFoundError(f"Critical error: image disappeared before copying: {source_path}")

        destination_path = output_folder / source_path.name
        shutil.copy2(source_path, destination_path)
        copied_paths.append(str(destination_path))

    split_df = split_df.copy()
    split_df["copied_path"] = copied_paths
    return split_df


def save_split_csv(split_df, output_csv_path):
    # حفظ ملف CSV الخاص بالتقسيم بالأعمدة المطلوبة فقط.
    columns = [
        "image_name",
        "image_path",
        "new_split",
        "quality_label",
        "readable_quality_label",
        "patient_id",
        "eye_side",
        "original_source_csv",
        "original_source_folder",
        "copied_path",
    ]

    split_df[columns].to_csv(output_csv_path, index=False)


def distribution_text(series):
    # تحويل توزيع الفئات إلى نص واضح للتقرير.
    counts = series.value_counts().sort_index()
    lines = []

    for label in [0, 1, 2]:
        readable_label = get_readable_quality_label(label)
        lines.append(f"{label} ({readable_label}): {counts.get(label, 0)}")

    return "\n".join(lines)


def save_overall_class_distribution(found_df):
    # رسم توزيع الفئات العام للصور الموجودة.
    counts = found_df["quality_label"].value_counts().sort_index()
    labels = [0, 1, 2]
    values = [counts.get(label, 0) for label in labels]

    plt.figure(figsize=(6, 4))
    plt.bar([str(label) for label in labels], values, color=["#2E7D32", "#1565C0", "#C62828"])
    plt.title("Overall Class Distribution")
    plt.xlabel("Quality label")
    plt.ylabel("Image count")
    plt.tight_layout()
    plt.savefig(OVERALL_CLASS_PLOT_PATH, dpi=150)
    plt.close()


def save_split_class_distribution(train_split, val_split, test_split):
    # رسم مقارنة توزيع الفئات بين train و val و test.
    labels = [0, 1, 2]
    x_positions = list(range(len(labels)))
    bar_width = 0.25

    split_names = ["train", "val", "test"]
    split_dataframes = [train_split, val_split, test_split]
    offsets = [-bar_width, 0, bar_width]

    plt.figure(figsize=(8, 4))

    for split_name, split_df, offset in zip(split_names, split_dataframes, offsets):
        counts = split_df["quality_label"].value_counts().sort_index()
        values = [counts.get(label, 0) for label in labels]
        bar_positions = [x + offset for x in x_positions]
        plt.bar(bar_positions, values, width=bar_width, label=split_name)

    plt.xticks(x_positions, [str(label) for label in labels])
    plt.title("Class Distribution by Split")
    plt.xlabel("Quality label")
    plt.ylabel("Image count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SPLIT_CLASS_PLOT_PATH, dpi=150)
    plt.close()


def save_split_percentages(train_count, val_count, test_count):
    # رسم النسب الفعلية للصور داخل كل تقسيم.
    total_count = train_count + val_count + test_count

    if total_count == 0:
        percentages = [0, 0, 0]
    else:
        percentages = [
            (train_count / total_count) * 100,
            (val_count / total_count) * 100,
            (test_count / total_count) * 100,
        ]

    plt.figure(figsize=(6, 4))
    plt.bar(["train", "val", "test"], percentages, color=["#2E7D32", "#1565C0", "#C62828"])
    plt.title("Actual Split Percentages")
    plt.ylabel("Percentage of images")
    plt.tight_layout()
    plt.savefig(SPLIT_PERCENTAGES_PLOT_PATH, dpi=150)
    plt.close()


def write_summary_report(
    train_csv_rows,
    test_csv_rows,
    master_df,
    found_df,
    train_split,
    val_split,
    test_split,
    leakage_passed,
):
    # كتابة تقرير نصي شامل بالمعلومات المطلوبة.
    total_split_images = len(train_split) + len(val_split) + len(test_split)

    if total_split_images == 0:
        train_percentage = 0
        val_percentage = 0
        test_percentage = 0
    else:
        train_percentage = (len(train_split) / total_split_images) * 100
        val_percentage = (len(val_split) / total_split_images) * 100
        test_percentage = (len(test_split) / total_split_images) * 100

    output_files = [
        MASTER_MANIFEST_PATH,
        TRAIN_SPLIT_CSV_PATH,
        VAL_SPLIT_CSV_PATH,
        TEST_SPLIT_CSV_PATH,
        SUMMARY_REPORT_PATH,
        OVERALL_CLASS_PLOT_PATH,
        SPLIT_CLASS_PLOT_PATH,
        SPLIT_PERCENTAGES_PLOT_PATH,
    ]

    lines = [
        "EyeQ Split Dataset Summary Report",
        "=" * 40,
        f"Total rows from Label_EyeQ_train.csv: {train_csv_rows}",
        f"Total rows from Label_EyeQ_test.csv: {test_csv_rows}",
        f"Total merged rows: {len(master_df)}",
        f"Number of found images: {len(found_df)}",
        f"Number of missing images: {len(master_df) - len(found_df)}",
        f"Total images used in final split: {total_split_images}",
        f"Number of unique patients: {found_df['patient_id'].nunique()}",
        f"Number of patients in train: {train_split['patient_id'].nunique()}",
        f"Number of patients in val: {val_split['patient_id'].nunique()}",
        f"Number of patients in test: {test_split['patient_id'].nunique()}",
        f"Number of images in train: {len(train_split)}",
        f"Number of images in val: {len(val_split)}",
        f"Number of images in test: {len(test_split)}",
        "",
        "Actual split percentages:",
        f"train: {train_percentage:.2f}%",
        f"val: {val_percentage:.2f}%",
        f"test: {test_percentage:.2f}%",
        "",
        "Class distribution overall:",
        distribution_text(found_df["quality_label"]),
        "",
        "Class distribution in train:",
        distribution_text(train_split["quality_label"]),
        "",
        "Class distribution in val:",
        distribution_text(val_split["quality_label"]),
        "",
        "Class distribution in test:",
        distribution_text(test_split["quality_label"]),
        "",
        f"Patient leakage check: {'PASSED' if leakage_passed else 'FAILED'}",
        "",
        "Output files created:",
    ]

    for output_file in output_files:
        lines.append(str(output_file))

    lines.extend(
        [
            "",
            "Libraries used and why:",
            "pathlib: clean path handling",
            "pandas: reading, merging, filtering, and saving CSV files",
            "shutil: copying images with metadata using copy2",
            "matplotlib: creating required visualizations",
            "sklearn.model_selection: patient-level train / val / test splitting",
            "collections: simple counting for representative patient labels",
        ]
    )

    SUMMARY_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    # تشغيل خطوات الدمج والتقسيم كاملة.
    print("Starting EyeQ master manifest and patient-safe split creation")

    create_output_folders()

    train_csv_df, train_manifest_df = build_manifest_from_csv(TRAIN_CSV_PATH)
    test_csv_df, test_manifest_df = build_manifest_from_csv(TEST_CSV_PATH)

    master_df = pd.concat([train_manifest_df, test_manifest_df], ignore_index=True)
    master_df.to_csv(MASTER_MANIFEST_PATH, index=False)
    print(f"Master manifest saved: {MASTER_MANIFEST_PATH}")

    found_df = master_df[master_df["found_status"] == "Found"].copy()
    missing_count = len(master_df) - len(found_df)

    print(f"Found images: {len(found_df)}")
    print(f"Missing images: {missing_count}")

    if found_df.empty:
        raise ValueError("Critical error: no found images are available for splitting.")

    validate_quality_labels(found_df)

    print("Creating patient-level train / val / test split")
    train_split, val_split, test_split = split_patients(found_df)

    leakage_passed = check_patient_leakage(train_split, val_split, test_split)

    if not leakage_passed:
        raise ValueError("Critical error: patient leakage was detected after splitting.")

    print("Patient leakage check passed")

    print("Copying train images")
    train_split = copy_split_images(train_split, OUTPUT_TRAIN_DIR)

    print("Copying validation images")
    val_split = copy_split_images(val_split, OUTPUT_VAL_DIR)

    print("Copying test images")
    test_split = copy_split_images(test_split, OUTPUT_TEST_DIR)

    print("Saving split CSV files")
    save_split_csv(train_split, TRAIN_SPLIT_CSV_PATH)
    save_split_csv(val_split, VAL_SPLIT_CSV_PATH)
    save_split_csv(test_split, TEST_SPLIT_CSV_PATH)

    print("Creating visualizations")
    save_overall_class_distribution(found_df)
    save_split_class_distribution(train_split, val_split, test_split)
    save_split_percentages(len(train_split), len(val_split), len(test_split))

    print("Writing summary report")
    write_summary_report(
        train_csv_rows=len(train_csv_df),
        test_csv_rows=len(test_csv_df),
        master_df=master_df,
        found_df=found_df,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        leakage_passed=leakage_passed,
    )

    print("\nFinal Summary")
    print("=" * 40)
    print(f"Master manifest path: {MASTER_MANIFEST_PATH}")
    print(f"Train split CSV path: {TRAIN_SPLIT_CSV_PATH}")
    print(f"Validation split CSV path: {VAL_SPLIT_CSV_PATH}")
    print(f"Test split CSV path: {TEST_SPLIT_CSV_PATH}")
    print(f"Report path: {SUMMARY_REPORT_PATH}")
    print(f"Train image count: {len(train_split)}")
    print(f"Validation image count: {len(val_split)}")
    print(f"Test image count: {len(test_split)}")
    print(f"Patient leakage check: {'PASSED' if leakage_passed else 'FAILED'}")


if __name__ == "__main__":
    main()