from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# هذا السكربت يجهز صور EyeQ للتدريب عن طريق تحويلها إلى RGB وتغيير حجمها إلى 512x512.
# يستخدم resize مع الحفاظ على نسبة الأبعاد ثم يضيف padding أسود حتى لا يتم تشويه الصورة الطبية.
# لا يقوم هذا السكربت بتدريب أي نموذج، ولا يعدل الصور الأصلية، ولا يطبق augmentation، ولا يغير التسميات.
# تبقى البيانات ثلاث فئات كما هي: 0 = Good، 1 = Usable، 2 = Reject.
# المكتبات المستخدمة: pathlib للمسارات، pandas لملفات CSV، Pillow لقراءة وحفظ الصور، matplotlib للرسومات، collections للعد البسيط.


DATASET_ROOT = Path(r"C:\EyePACS\EyeQ_Split_Dataset")
INPUT_TRAIN_DIR = DATASET_ROOT / "train"
INPUT_VAL_DIR = DATASET_ROOT / "val"
INPUT_TEST_DIR = DATASET_ROOT / "test"

INPUT_REPORTS_DIR = DATASET_ROOT / "reports"
TRAIN_CSV_PATH = INPUT_REPORTS_DIR / "train_split.csv"
VAL_CSV_PATH = INPUT_REPORTS_DIR / "val_split.csv"
TEST_CSV_PATH = INPUT_REPORTS_DIR / "test_split.csv"

OUTPUT_ROOT = Path(r"C:\EyePACS\EyeQ_Preprocessed_512")
OUTPUT_TRAIN_DIR = OUTPUT_ROOT / "train"
OUTPUT_VAL_DIR = OUTPUT_ROOT / "val"
OUTPUT_TEST_DIR = OUTPUT_ROOT / "test"
OUTPUT_REPORTS_DIR = OUTPUT_ROOT / "reports"

TRAIN_OUTPUT_CSV_PATH = OUTPUT_REPORTS_DIR / "train_preprocessed.csv"
VAL_OUTPUT_CSV_PATH = OUTPUT_REPORTS_DIR / "val_preprocessed.csv"
TEST_OUTPUT_CSV_PATH = OUTPUT_REPORTS_DIR / "test_preprocessed.csv"
SUMMARY_REPORT_PATH = OUTPUT_REPORTS_DIR / "preprocessing_summary_report.txt"
COUNTS_PLOT_PATH = OUTPUT_REPORTS_DIR / "preprocessing_counts.png"
SIZE_SUMMARY_PLOT_PATH = OUTPUT_REPORTS_DIR / "before_after_size_summary.png"

TARGET_SIZE = 512
SUPPORTED_EXTENSIONS = [".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"]

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
    OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def read_split_csv(csv_path):
    # قراءة ملف CSV الخاص بالتقسيم مع إظهار رسالة واضحة إذا كان مفقودا.
    if not csv_path.exists():
        raise FileNotFoundError(f"Critical error: split CSV file was not found: {csv_path}")

    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Rows found: {len(df)}")
    print(f"Columns found: {list(df.columns)}")

    return df


def get_value(row, column_name, default_value=""):
    # قراءة قيمة من صف CSV إذا كان العمود موجودا.
    if column_name in row.index:
        return row[column_name]
    return default_value


def clean_image_name(value):
    # تنظيف اسم الصورة وإزالة أي مسار أو امتداد إن وجد.
    return Path(str(value).strip()).stem


def get_readable_quality_label(row):
    # استخدام readable_quality_label من CSV إن وجد، وإلا إنشاؤه من quality_label.
    existing_label = get_value(row, "readable_quality_label", "")

    if pd.notna(existing_label) and str(existing_label).strip():
        return existing_label

    quality_label = get_value(row, "quality_label", "")

    try:
        quality_label = int(quality_label)
    except (TypeError, ValueError):
        return ""

    return QUALITY_LABEL_MAP.get(quality_label, "")


def detect_image_path_from_row(row):
    # استخدام عمود مسار الصورة إذا كان موجودا وصالحا.
    candidate_columns = ["image_path", "copied_path", "original_image_path"]

    for column_name in candidate_columns:
        if column_name not in row.index:
            continue

        value = row[column_name]

        if pd.isna(value):
            continue

        candidate_path = Path(str(value).strip())

        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path

    return None


def search_image_in_split_folder(image_name, split_folder):
    # البحث عن الصورة داخل مجلد التقسيم إذا كان المسار في CSV مفقودا أو غير صالح.
    image_stem = clean_image_name(image_name)

    for extension in SUPPORTED_EXTENSIONS:
        candidate_path = split_folder / f"{image_stem}{extension}"
        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path

    return None


def find_input_image(row, split_folder):
    # تحديد مسار الصورة من CSV أولا، ثم البحث داخل مجلد التقسيم.
    image_path = detect_image_path_from_row(row)

    if image_path is not None:
        return image_path

    image_name = get_value(row, "image_name", "")

    if pd.isna(image_name) or not str(image_name).strip():
        return None

    return search_image_in_split_folder(image_name, split_folder)


def resize_with_black_padding(image, target_size):
    # تغيير الحجم مع الحفاظ على نسبة الأبعاد ثم إضافة padding أسود للوصول إلى 512x512.
    original_width, original_height = image.size

    scale = target_size / max(original_width, original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))

    resized_image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    padded_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - resized_width) // 2
    paste_y = (target_size - resized_height) // 2
    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image


def preprocess_one_image(input_image_path, output_image_path):
    # قراءة صورة واحدة وتحويلها إلى RGB ثم حفظها بالحجم النهائي 512x512.
    with Image.open(input_image_path) as image:
        rgb_image = image.convert("RGB")
        original_width, original_height = rgb_image.size
        processed_image = resize_with_black_padding(rgb_image, TARGET_SIZE)
        processed_image.save(output_image_path, format="JPEG", quality=95)

    return original_width, original_height


def process_split(split_name, input_csv_path, input_folder, output_folder, output_csv_path):
    # معالجة كل صور تقسيم واحد وحفظ CSV جديد بنتائج المعالجة.
    print(f"\nProcessing split: {split_name}")

    df = read_split_csv(input_csv_path)
    rows = []

    success_count = 0
    failed_count = 0
    original_sizes = []

    for index, row in df.iterrows():
        image_name_value = get_value(row, "image_name", "")

        if pd.isna(image_name_value) or not str(image_name_value).strip():
            image_name = ""
        else:
            image_name = clean_image_name(image_name_value)

        quality_label = get_value(row, "quality_label", "")
        readable_quality_label = get_readable_quality_label(row)
        patient_id = get_value(row, "patient_id", "")
        eye_side = get_value(row, "eye_side", "")

        input_image_path = find_input_image(row, input_folder)

        original_image_path = ""
        preprocessed_image_path = ""
        original_width = ""
        original_height = ""
        original_size = ""
        new_width = ""
        new_height = ""
        new_size = ""
        preprocessing_status = "Failed"
        error_message = ""

        try:
            if input_image_path is None:
                raise FileNotFoundError(f"Image file was not found for image_name: {image_name}")

            original_image_path = str(input_image_path)
            output_image_path = output_folder / f"{clean_image_name(input_image_path.name)}.jpeg"

            original_width, original_height = preprocess_one_image(input_image_path, output_image_path)

            original_size = f"{original_width}x{original_height}"
            new_width = TARGET_SIZE
            new_height = TARGET_SIZE
            new_size = f"{TARGET_SIZE}x{TARGET_SIZE}"
            preprocessed_image_path = str(output_image_path)
            preprocessing_status = "Success"
            success_count += 1
            original_sizes.append(original_size)

        except Exception as error:
            failed_count += 1
            error_message = str(error)

        rows.append(
            {
                "image_name": image_name,
                "quality_label": quality_label,
                "readable_quality_label": readable_quality_label,
                "patient_id": patient_id,
                "eye_side": eye_side,
                "original_image_path": original_image_path,
                "preprocessed_image_path": preprocessed_image_path,
                "original_width": original_width,
                "original_height": original_height,
                "original_size": original_size,
                "new_width": new_width,
                "new_height": new_height,
                "new_size": new_size,
                "preprocessing_status": preprocessing_status,
                "error_message": error_message,
            }
        )

        if (index + 1) % 500 == 0 or (index + 1) == len(df):
            print(f"{split_name}: processed {index + 1} / {len(df)}")

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_csv_path, index=False)

    print(f"{split_name}: successful = {success_count}, failed = {failed_count}")
    print(f"Saved CSV: {output_csv_path}")

    return {
        "split_name": split_name,
        "total_images": len(df),
        "success_count": success_count,
        "failed_count": failed_count,
        "original_sizes": original_sizes,
        "output_csv_path": output_csv_path,
    }


def get_most_common_original_sizes(results):
    # حساب أكثر الأحجام الأصلية شيوعا قبل المعالجة.
    all_sizes = []

    for result in results.values():
        all_sizes.extend(result["original_sizes"])

    size_counts = Counter(all_sizes)
    return size_counts.most_common(10)


def save_preprocessing_counts_plot(results):
    # رسم عدد الصور الناجحة والفاشلة لكل تقسيم.
    split_names = ["train", "val", "test"]
    success_values = [results[split_name]["success_count"] for split_name in split_names]
    failed_values = [results[split_name]["failed_count"] for split_name in split_names]

    x_positions = list(range(len(split_names)))
    bar_width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar([x - bar_width / 2 for x in x_positions], success_values, width=bar_width, label="Success")
    plt.bar([x + bar_width / 2 for x in x_positions], failed_values, width=bar_width, label="Failed")
    plt.xticks(x_positions, split_names)
    plt.title("Preprocessing Counts")
    plt.xlabel("Split")
    plt.ylabel("Image count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(COUNTS_PLOT_PATH, dpi=150)
    plt.close()


def save_before_after_size_summary_plot(results):
    # رسم أكثر الأحجام الأصلية شيوعا مع توضيح أن الحجم النهائي ثابت 512x512.
    most_common_sizes = get_most_common_original_sizes(results)

    labels = [size for size, _ in most_common_sizes]
    values = [count for _, count in most_common_sizes]

    total_success = sum(result["success_count"] for result in results.values())

    labels.append("512x512 after")
    values.append(total_success)

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color="#1565C0")
    plt.title("Before and After Size Summary")
    plt.xlabel("Image size")
    plt.ylabel("Image count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(SIZE_SUMMARY_PLOT_PATH, dpi=150)
    plt.close()


def write_summary_report(results):
    # كتابة تقرير نصي شامل عن عملية المعالجة.
    total_processed = sum(result["success_count"] for result in results.values())
    total_failed = sum(result["failed_count"] for result in results.values())

    lines = [
        "EyeQ 512x512 Preprocessing Summary Report",
        "=" * 50,
        f"Target image size: {TARGET_SIZE}x{TARGET_SIZE}",
        "Preprocessing method: aspect-ratio resize + black padding",
        "",
        f"Total train images: {results['train']['total_images']}",
        f"Successfully processed train images: {results['train']['success_count']}",
        f"Failed train images: {results['train']['failed_count']}",
        "",
        f"Total val images: {results['val']['total_images']}",
        f"Successfully processed val images: {results['val']['success_count']}",
        f"Failed val images: {results['val']['failed_count']}",
        "",
        f"Total test images: {results['test']['total_images']}",
        f"Successfully processed test images: {results['test']['success_count']}",
        f"Failed test images: {results['test']['failed_count']}",
        "",
        f"Total processed images: {total_processed}",
        f"Total failed images: {total_failed}",
        "",
        "Confirmation that original images were not modified: Yes",
        "Confirmation that labels were not changed: Yes",
        "Confirmation that no augmentation was applied: Yes",
        "",
        "Libraries used and why:",
        "pathlib: clean path handling",
        "pandas: reading and saving CSV reports",
        "Pillow: reading images, converting to RGB, resizing with padding, and saving JPEG files",
        "matplotlib: creating simple required visualizations",
        "collections: counting original image sizes",
    ]

    SUMMARY_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {SUMMARY_REPORT_PATH}")


def verify_successful_outputs_are_512(output_csv_paths):
    # التأكد من أن كل الصور الناجحة في CSV أصبحت 512x512.
    for csv_path in output_csv_paths:
        df = pd.read_csv(csv_path)
        success_df = df[df["preprocessing_status"] == "Success"]

        for _, row in success_df.iterrows():
            output_path = Path(row["preprocessed_image_path"])

            if not output_path.exists():
                return False

            with Image.open(output_path) as image:
                if image.size != (TARGET_SIZE, TARGET_SIZE):
                    return False

    return True


def main():
    # تشغيل خطوات المعالجة كاملة.
    print("Starting EyeQ 512x512 preprocessing")
    create_output_folders()

    results = {}

    results["train"] = process_split(
        split_name="train",
        input_csv_path=TRAIN_CSV_PATH,
        input_folder=INPUT_TRAIN_DIR,
        output_folder=OUTPUT_TRAIN_DIR,
        output_csv_path=TRAIN_OUTPUT_CSV_PATH,
    )

    results["val"] = process_split(
        split_name="val",
        input_csv_path=VAL_CSV_PATH,
        input_folder=INPUT_VAL_DIR,
        output_folder=OUTPUT_VAL_DIR,
        output_csv_path=VAL_OUTPUT_CSV_PATH,
    )

    results["test"] = process_split(
        split_name="test",
        input_csv_path=TEST_CSV_PATH,
        input_folder=INPUT_TEST_DIR,
        output_folder=OUTPUT_TEST_DIR,
        output_csv_path=TEST_OUTPUT_CSV_PATH,
    )

    print("\nCreating visualizations")
    save_preprocessing_counts_plot(results)
    save_before_after_size_summary_plot(results)

    print("Writing summary report")
    write_summary_report(results)

    output_check_passed = verify_successful_outputs_are_512(
        [
            TRAIN_OUTPUT_CSV_PATH,
            VAL_OUTPUT_CSV_PATH,
            TEST_OUTPUT_CSV_PATH,
        ]
    )

    total_failed = sum(result["failed_count"] for result in results.values())

    print("\nFinal Summary")
    print("=" * 40)
    print(f"Output train folder path: {OUTPUT_TRAIN_DIR}")
    print(f"Output val folder path: {OUTPUT_VAL_DIR}")
    print(f"Output test folder path: {OUTPUT_TEST_DIR}")
    print(f"Reports folder path: {OUTPUT_REPORTS_DIR}")
    print(f"Successfully processed train images: {results['train']['success_count']}")
    print(f"Successfully processed val images: {results['val']['success_count']}")
    print(f"Successfully processed test images: {results['test']['success_count']}")
    print(f"Number of failed images: {total_failed}")
    print(f"All successful output images are 512x512: {'Yes' if output_check_passed else 'No'}")


if __name__ == "__main__":
    main()