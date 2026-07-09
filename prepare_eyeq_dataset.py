# هذا السكربت يراجع ويجهز بيانات EyeQ / EyePACS لجودة صور الشبكية.
# يستخدم ملفات Label_EyeQ_train.csv و Label_EyeQ_test.csv لمعرفة أسماء الصور والتسميات إن وجدت.
# ينسخ الصور الموجودة فقط إلى Final Train و Final Test بدون تعديل الصور الأصلية أو إعادة تسميتها.
# ينشئ تقارير CSV وتقرير نصي ورسومات مهمة داخل مجلد reports.
# المكتبات المستخدمة: pathlib للمسارات، pandas لقراءة CSV، shutil لنسخ الصور، matplotlib للرسم.

from pathlib import Path
import shutil

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(r"C:\EyePACS")

TRAIN_IMAGES_DIR = BASE_DIR / "train"
TEST_IMAGES_DIR = BASE_DIR / "test"

TRAIN_CSV_PATH = BASE_DIR / "Label_EyeQ_train.csv"
TEST_CSV_PATH = BASE_DIR / "Label_EyeQ_test.csv"

FINAL_TRAIN_DIR = BASE_DIR / "Final Train"
FINAL_TEST_DIR = BASE_DIR / "Final Test"
REPORTS_DIR = BASE_DIR / "reports"

SUPPORTED_EXTENSIONS = [".jpeg", ".jpg", ".png"]


def detect_image_column(columns):
    # نحاول اكتشاف عمود اسم الصورة من أسماء الأعمدة الشائعة.
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
    # نحاول اكتشاف عمود تصنيف الجودة بدون افتراض اسم محدد.
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
        column_lower = column.lower().strip()
        if column == image_column:
            continue
        if "quality" in column_lower or "label" in column_lower or "class" in column_lower:
            return column

    return None


def clean_image_name(value):
    # نحول اسم الصورة إلى نص ونزيل المسافات والامتداد إن كان موجودا.
    image_name = str(value).strip()
    return Path(image_name).stem


def find_image_file(image_name):
    # نبحث عن الصورة في مجلدي train و test وبالامتدادات المدعومة فقط.
    image_stem = clean_image_name(image_name)

    for source_name, source_dir in [("train", TRAIN_IMAGES_DIR), ("test", TEST_IMAGES_DIR)]:
        for extension in SUPPORTED_EXTENSIONS:
            candidate_path = source_dir / f"{image_stem}{extension}"
            if candidate_path.exists():
                return source_name, candidate_path

    return None, None


def copy_found_image(original_path, final_dir):
    # ننسخ الصورة كما هي مع الاحتفاظ باسمها وبياناتها الوصفية.
    final_path = final_dir / original_path.name
    shutil.copy2(original_path, final_path)
    return final_path


def safe_percentage(part, total):
    # نحسب النسبة مع تجنب القسمة على صفر.
    if total == 0:
        return 0.0
    return (part / total) * 100


def format_distribution(series):
    # نحول توزيع الفئات إلى نص واضح داخل التقرير.
    if series is None:
        return "Quality label column was not detected."

    if series.empty:
        return "No found images available for class distribution."

    counts = series.value_counts(dropna=False).sort_index()
    lines = []
    for label, count in counts.items():
        lines.append(f"{label}: {count}")
    return "\n".join(lines)


def save_found_missing_chart(found_count, missing_count, title, output_path):
    # نحفظ رسم يوضح عدد الصور الموجودة والمفقودة.
    plt.figure(figsize=(6, 4))
    plt.bar(["Found", "Missing"], [found_count, missing_count], color=["#2E7D32", "#C62828"])
    plt.title(title)
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_class_distribution_chart(series, title, output_path):
    # نحفظ رسم توزيع فئات الجودة بعد الاحتفاظ بالصور الموجودة فقط.
    if series is None or series.empty:
        return

    counts = series.value_counts(dropna=False).sort_index()

    plt.figure(figsize=(7, 4))
    plt.bar(counts.index.astype(str), counts.values, color="#1565C0")
    plt.title(title)
    plt.xlabel("Quality class")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_combined_class_distribution(train_series, test_series, output_path):
    # نحفظ رسم مقارنة بين توزيع فئات Train و Test بعد الفلترة.
    if train_series is None or test_series is None:
        return

    if train_series.empty and test_series.empty:
        return

    train_counts = train_series.value_counts(dropna=False)
    test_counts = test_series.value_counts(dropna=False)

    all_labels = sorted(set(train_counts.index.astype(str)) | set(test_counts.index.astype(str)))

    train_values = [train_counts.get(label, 0) for label in all_labels]
    test_values = [test_counts.get(label, 0) for label in all_labels]

    x_positions = range(len(all_labels))
    bar_width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar([x - bar_width / 2 for x in x_positions], train_values, width=bar_width, label="Train")
    plt.bar([x + bar_width / 2 for x in x_positions], test_values, width=bar_width, label="Test")
    plt.xticks(list(x_positions), all_labels)
    plt.title("Train vs Test Quality Class Distribution")
    plt.xlabel("Quality class")
    plt.ylabel("Number of images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def audit_dataset(csv_path, final_dir, audit_csv_path, dataset_name):
    # نقرأ ملف CSV ونفحص الأعمدة قبل استخدامها.
    print(f"\nReading {csv_path}")

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    print(f"Columns found in {csv_path.name}: {list(df.columns)}")

    image_column = detect_image_column(df.columns)
    quality_column = detect_quality_column(df.columns, image_column)

    if image_column is None:
        print(f"ERROR: Could not detect image-name column in {csv_path.name}")
        return None

    if quality_column is None:
        print(f"WARNING: Could not detect quality-label column in {csv_path.name}")

    total_rows = len(df)
    found_count = 0
    missing_count = 0
    found_in_train_count = 0
    found_in_test_count = 0

    audit_rows = []
    found_quality_labels = []

    print(f"Auditing {total_rows} rows from {csv_path.name}")

    for index, row in df.iterrows():
        image_name = row[image_column]
        quality_label = row[quality_column] if quality_column is not None else ""

        original_location, original_path = find_image_file(image_name)

        if original_path is not None:
            final_path = copy_found_image(original_path, final_dir)
            found_status = "found"
            found_count += 1

            if original_location == "train":
                found_in_train_count += 1
            elif original_location == "test":
                found_in_test_count += 1

            if quality_column is not None:
                found_quality_labels.append(quality_label)
        else:
            final_path = ""
            found_status = "missing"
            original_location = ""
            original_path = ""
            missing_count += 1

        audit_rows.append(
            {
                "image_name": image_name,
                "quality_label": quality_label,
                "found_status": found_status,
                "original_location": original_location,
                "original_path": str(original_path) if original_path else "",
                "final_path": str(final_path) if final_path else "",
            }
        )

        if (index + 1) % 500 == 0 or (index + 1) == total_rows:
            print(f"{dataset_name}: processed {index + 1} / {total_rows}")

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(audit_csv_path, index=False)

    class_before = df[quality_column] if quality_column is not None else None
    class_after = pd.Series(found_quality_labels) if quality_column is not None else None

    result = {
        "dataset_name": dataset_name,
        "csv_path": csv_path,
        "total_rows": total_rows,
        "found_count": found_count,
        "missing_count": missing_count,
        "found_percentage": safe_percentage(found_count, total_rows),
        "missing_percentage": safe_percentage(missing_count, total_rows),
        "class_before": class_before,
        "class_after": class_after,
        "found_in_train_count": found_in_train_count,
        "found_in_test_count": found_in_test_count,
        "audit_csv_path": audit_csv_path,
        "quality_column": quality_column,
    }

    print(f"{dataset_name}: found {found_count}, missing {missing_count}")
    return result


def write_summary_report(train_result, test_result, report_path):
    # نكتب تقريرا نصيا كاملا يحتوي على الأرقام والتوزيعات المطلوبة.
    lines = []

    for result in [train_result, test_result]:
        if result is None:
            continue

        lines.append(f"{result['dataset_name']} CSV Report")
        lines.append("=" * 40)
        lines.append(f"CSV path: {result['csv_path']}")
        lines.append(f"Total rows in CSV: {result['total_rows']}")
        lines.append(f"Number of found images: {result['found_count']}")
        lines.append(f"Number of missing images: {result['missing_count']}")
        lines.append(f"Percentage found: {result['found_percentage']:.2f}%")
        lines.append(f"Percentage missing: {result['missing_percentage']:.2f}%")
        lines.append(f"Number of images found in original train folder: {result['found_in_train_count']}")
        lines.append(f"Number of images found in original test folder: {result['found_in_test_count']}")
        lines.append("")
        lines.append("Class distribution before filtering:")
        lines.append(format_distribution(result["class_before"]))
        lines.append("")
        lines.append("Class distribution after keeping only found images:")
        lines.append(format_distribution(result["class_after"]))
        lines.append("")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    # ننشئ المجلدات النهائية المطلوبة فقط.
    FINAL_TRAIN_DIR.mkdir(exist_ok=True)
    FINAL_TEST_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Starting EyeQ / EyePACS audit and preparation")
    print(f"Base folder: {BASE_DIR}")

    train_result = audit_dataset(
        csv_path=TRAIN_CSV_PATH,
        final_dir=FINAL_TRAIN_DIR,
        audit_csv_path=REPORTS_DIR / "train_audit.csv",
        dataset_name="Train",
    )

    test_result = audit_dataset(
        csv_path=TEST_CSV_PATH,
        final_dir=FINAL_TEST_DIR,
        audit_csv_path=REPORTS_DIR / "test_audit.csv",
        dataset_name="Test",
    )

    if train_result is not None:
        save_found_missing_chart(
            train_result["found_count"],
            train_result["missing_count"],
            "Train CSV: Found vs Missing",
            REPORTS_DIR / "train_found_missing.png",
        )
        save_class_distribution_chart(
            train_result["class_after"],
            "Train Quality Class Distribution After Filtering",
            REPORTS_DIR / "train_class_distribution.png",
        )

    if test_result is not None:
        save_found_missing_chart(
            test_result["found_count"],
            test_result["missing_count"],
            "Test CSV: Found vs Missing",
            REPORTS_DIR / "test_found_missing.png",
        )
        save_class_distribution_chart(
            test_result["class_after"],
            "Test Quality Class Distribution After Filtering",
            REPORTS_DIR / "test_class_distribution.png",
        )

    if train_result is not None and test_result is not None:
        save_combined_class_distribution(
            train_result["class_after"],
            test_result["class_after"],
            REPORTS_DIR / "combined_class_distribution.png",
        )

    write_summary_report(
        train_result=train_result,
        test_result=test_result,
        report_path=REPORTS_DIR / "final_summary_report.txt",
    )

    train_copied = train_result["found_count"] if train_result is not None else 0
    test_copied = test_result["found_count"] if test_result is not None else 0
    train_missing = train_result["missing_count"] if train_result is not None else 0
    test_missing = test_result["missing_count"] if test_result is not None else 0

    print("\nFinal Summary")
    print("=" * 40)
    print(f"Final Train folder path: {FINAL_TRAIN_DIR}")
    print(f"Final Test folder path: {FINAL_TEST_DIR}")
    print(f"Reports folder path: {REPORTS_DIR}")
    print(f"Images copied to Final Train: {train_copied}")
    print(f"Images copied to Final Test: {test_copied}")
    print(f"Missing images from train CSV: {train_missing}")
    print(f"Missing images from test CSV: {test_missing}")
    print("Done.")


if __name__ == "__main__":
    main()