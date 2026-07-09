from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# هذا السكربت مخصص فقط لعمل بروفايل لصور Dataset بعد تقسيم EyeQ.
# يفحص امتدادات الصور وأبعادها داخل مجلدات train و val و test.
# لا يقوم بتغيير البيانات ولا ينسخ ولا ينقل ولا يحذف ولا يعيد تسمية ولا يغير حجم أي صورة.
# يساعد هذا التقرير في اختيار استراتيجية resize مناسبة قبل التدريب لاحقا.
# المكتبات المستخدمة: pathlib للمسارات، pandas للجداول وتقارير CSV، Pillow لقراءة أبعاد الصور، matplotlib للرسومات.


BASE_DIR = Path(r"C:\EyePACS\EyeQ_Split_Dataset")

SPLIT_FOLDERS = {
    "train": BASE_DIR / "train",
    "val": BASE_DIR / "val",
    "test": BASE_DIR / "test",
}

REPORTS_DIR = BASE_DIR / "reports" / "image_profile"

SUPPORTED_EXTENSIONS = [".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"]


def create_reports_folder():
    # إنشاء مجلد التقارير فقط إذا لم يكن موجودا.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def find_image_files(folder_path):
    # جمع ملفات الصور المدعومة من المجلد بدون تعديل أي ملف.
    if not folder_path.exists():
        print(f"Warning: folder does not exist: {folder_path}")
        return []

    image_files = []

    for path in folder_path.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(path)

    return sorted(image_files)


def read_image_info(image_path):
    # قراءة أبعاد الصورة باستخدام Pillow بدون تعديل الصورة.
    try:
        with Image.open(image_path) as image:
            width, height = image.size

        return width, height, "Readable", ""

    except Exception as error:
        return "", "", "Unreadable", str(error)


def profile_split(split_name, folder_path):
    # عمل بروفايل كامل لمجلد واحد من مجلدات train أو val أو test.
    print(f"\nProfiling split: {split_name}")
    print(f"Folder: {folder_path}")

    image_files = find_image_files(folder_path)
    total_images = len(image_files)

    print(f"Found {total_images} supported image files in {split_name}")

    rows = []

    for index, image_path in enumerate(image_files, start=1):
        width, height, readable_status, error_message = read_image_info(image_path)

        if readable_status == "Readable":
            image_size = f"{width}x{height}"
        else:
            image_size = ""

        rows.append(
            {
                "split_name": split_name,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "extension": image_path.suffix.lower(),
                "width": width,
                "height": height,
                "image_size": image_size,
                "readable_status": readable_status,
                "error_message": error_message,
            }
        )

        if index % 500 == 0 or index == total_images:
            print(f"{split_name}: processed {index} / {total_images}")

    return pd.DataFrame(rows)


def calculate_extension_distribution(profile_df, split_name):
    # حساب توزيع الامتدادات في مجلد واحد أو في البيانات المجمعة.
    total_images = len(profile_df)
    counts = profile_df["extension"].value_counts().sort_index()

    rows = []

    for extension, image_count in counts.items():
        percentage = (image_count / total_images) * 100 if total_images else 0

        rows.append(
            {
                "split_name": split_name,
                "extension": extension,
                "image_count": image_count,
                "percentage": percentage,
            }
        )

    return pd.DataFrame(rows)


def calculate_size_distribution(profile_df, split_name):
    # حساب توزيع أحجام الصور المقروءة فقط.
    readable_df = profile_df[profile_df["readable_status"] == "Readable"].copy()
    total_readable = len(readable_df)

    counts = readable_df["image_size"].value_counts()

    rows = []

    for image_size, image_count in counts.items():
        width_text, height_text = image_size.split("x")
        percentage = (image_count / total_readable) * 100 if total_readable else 0

        rows.append(
            {
                "split_name": split_name,
                "image_size": image_size,
                "width": int(width_text),
                "height": int(height_text),
                "image_count": image_count,
                "percentage": percentage,
            }
        )

    size_df = pd.DataFrame(rows)

    if not size_df.empty:
        size_df = size_df.sort_values(["image_count", "image_size"], ascending=[False, True])

    return size_df


def get_profile_summary(profile_df, extension_df, size_df):
    # استخراج ملخص الأرقام المهمة من profile dataframe.
    readable_df = profile_df[profile_df["readable_status"] == "Readable"].copy()
    unreadable_df = profile_df[profile_df["readable_status"] == "Unreadable"].copy()

    if readable_df.empty:
        min_width = ""
        max_width = ""
        min_height = ""
        max_height = ""
    else:
        min_width = int(readable_df["width"].min())
        max_width = int(readable_df["width"].max())
        min_height = int(readable_df["height"].min())
        max_height = int(readable_df["height"].max())

    if size_df.empty:
        most_common_size = ""
    else:
        most_common_size = size_df.iloc[0]["image_size"]

    if extension_df.empty:
        most_common_extension = ""
    else:
        sorted_extensions = extension_df.sort_values(["image_count", "extension"], ascending=[False, True])
        most_common_extension = sorted_extensions.iloc[0]["extension"]

    return {
        "total_images": len(profile_df),
        "unreadable_count": len(unreadable_df),
        "unreadable_paths": unreadable_df["image_path"].tolist(),
        "min_width": min_width,
        "max_width": max_width,
        "min_height": min_height,
        "max_height": max_height,
        "most_common_size": most_common_size,
        "most_common_extension": most_common_extension,
    }


def save_profile_csv(profile_df, split_name):
    # حفظ CSV يحتوي على معلومات كل صورة.
    output_path = REPORTS_DIR / f"{split_name}_image_profile.csv"
    profile_df.to_csv(output_path, index=False)
    return output_path


def save_extension_distribution_csv(extension_df, split_name):
    # حفظ CSV لتوزيع الامتدادات.
    output_path = REPORTS_DIR / f"{split_name}_extension_distribution.csv"
    extension_df.to_csv(output_path, index=False)
    return output_path


def save_size_distribution_csv(size_df, split_name):
    # حفظ CSV لتوزيع أحجام الصور.
    output_path = REPORTS_DIR / f"{split_name}_size_distribution.csv"
    size_df.to_csv(output_path, index=False)
    return output_path


def save_extension_plot(extension_df, split_name):
    # رسم توزيع الامتدادات لكل الصور.
    output_path = REPORTS_DIR / f"{split_name}_extension_distribution.png"

    plt.figure(figsize=(7, 4))

    if extension_df.empty:
        plt.bar(["No images"], [0])
    else:
        plt.bar(extension_df["extension"], extension_df["image_count"], color="#1565C0")

    plt.title(f"{split_name} Extension Distribution")
    plt.xlabel("Extension")
    plt.ylabel("Image count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def save_top_sizes_plot(size_df, split_name):
    # رسم أكثر 15 حجما تكرارا فقط حتى يبقى الرسم واضحا.
    output_path = REPORTS_DIR / f"{split_name}_top_image_sizes.png"

    top_sizes_df = size_df.head(15).copy()

    plt.figure(figsize=(10, 5))

    if top_sizes_df.empty:
        plt.bar(["No readable images"], [0])
    else:
        plt.bar(top_sizes_df["image_size"], top_sizes_df["image_count"], color="#2E7D32")
        plt.xticks(rotation=45, ha="right")

    plt.title(f"{split_name} Top Image Sizes")
    plt.xlabel("Image size")
    plt.ylabel("Image count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def format_extension_distribution(extension_df):
    # تحويل توزيع الامتدادات إلى نص للتقرير.
    if extension_df.empty:
        return "No supported image files found."

    lines = []

    for _, row in extension_df.iterrows():
        lines.append(
            f"{row['extension']}: {int(row['image_count'])} images ({row['percentage']:.2f}%)"
        )

    return "\n".join(lines)


def format_size_distribution(size_df, top_n=20):
    # تحويل أهم أحجام الصور إلى نص للتقرير.
    if size_df.empty:
        return "No readable images found."

    lines = []

    for _, row in size_df.head(top_n).iterrows():
        lines.append(
            f"{row['image_size']}: {int(row['image_count'])} images ({row['percentage']:.2f}%)"
        )

    return "\n".join(lines)


def format_unreadable_paths(unreadable_paths):
    # كتابة مسارات الصور غير المقروءة في التقرير إن وجدت.
    if not unreadable_paths:
        return "None"

    return "\n".join(unreadable_paths)


def write_summary_report(results):
    # كتابة تقرير نصي شامل لكل تقسيم وللبيانات المجمعة.
    report_lines = [
        "EyeQ Split Dataset Image Profile Report",
        "=" * 50,
        "",
    ]

    for split_name in ["train", "val", "test", "combined"]:
        result = results[split_name]
        summary = result["summary"]

        report_lines.extend(
            [
                f"{split_name.upper()}",
                "-" * 50,
                f"Total images: {summary['total_images']}",
                "",
                "Extension distribution:",
                format_extension_distribution(result["extension_df"]),
                "",
                "Size distribution top 20:",
                format_size_distribution(result["size_df"], top_n=20),
                "",
                f"Number of unreadable images: {summary['unreadable_count']}",
                "Unreadable image paths:",
                format_unreadable_paths(summary["unreadable_paths"]),
                "",
                f"Min width: {summary['min_width']}",
                f"Max width: {summary['max_width']}",
                f"Min height: {summary['min_height']}",
                f"Max height: {summary['max_height']}",
                f"Most common image size: {summary['most_common_size']}",
                f"Most common extension: {summary['most_common_extension']}",
                "",
            ]
        )

    report_lines.extend(
        [
            "Libraries used and why:",
            "pathlib: clean and reliable path handling",
            "pandas: tables and CSV reports",
            "Pillow: reading image dimensions without modifying images",
            "matplotlib: basic visualizations",
            "",
            "Dataset modification confirmation:",
            "This script did not modify, move, copy, delete, rename, or resize any image.",
        ]
    )

    output_path = REPORTS_DIR / "image_profile_summary_report.txt"
    output_path.write_text("\n".join(report_lines), encoding="utf-8")

    return output_path


def process_split(split_name, folder_path):
    # تنفيذ كل خطوات البروفايل لتقسيم واحد.
    profile_df = profile_split(split_name, folder_path)
    extension_df = calculate_extension_distribution(profile_df, split_name)
    size_df = calculate_size_distribution(profile_df, split_name)
    summary = get_profile_summary(profile_df, extension_df, size_df)

    save_profile_csv(profile_df, split_name)
    save_extension_distribution_csv(extension_df, split_name)
    save_size_distribution_csv(size_df, split_name)
    save_extension_plot(extension_df, split_name)
    save_top_sizes_plot(size_df, split_name)

    return {
        "profile_df": profile_df,
        "extension_df": extension_df,
        "size_df": size_df,
        "summary": summary,
    }


def process_combined(all_profile_dfs):
    # إنشاء ملخص مجمع لكل الصور من train و val و test.
    if all_profile_dfs:
        combined_profile_df = pd.concat(all_profile_dfs, ignore_index=True)
    else:
        combined_profile_df = pd.DataFrame(
            columns=[
                "split_name",
                "image_name",
                "image_path",
                "extension",
                "width",
                "height",
                "image_size",
                "readable_status",
                "error_message",
            ]
        )

    extension_df = calculate_extension_distribution(combined_profile_df, "combined")
    size_df = calculate_size_distribution(combined_profile_df, "combined")
    summary = get_profile_summary(combined_profile_df, extension_df, size_df)

    save_profile_csv(combined_profile_df, "combined")
    save_extension_distribution_csv(extension_df, "combined")
    save_size_distribution_csv(size_df, "combined")
    save_extension_plot(extension_df, "combined")
    save_top_sizes_plot(size_df, "combined")

    return {
        "profile_df": combined_profile_df,
        "extension_df": extension_df,
        "size_df": size_df,
        "summary": summary,
    }


def main():
    # نقطة التشغيل الرئيسية للسكريبت.
    print("Starting EyeQ split image profiling")
    create_reports_folder()

    results = {}
    all_profile_dfs = []

    for split_name, folder_path in SPLIT_FOLDERS.items():
        result = process_split(split_name, folder_path)
        results[split_name] = result
        all_profile_dfs.append(result["profile_df"])

    results["combined"] = process_combined(all_profile_dfs)

    summary_report_path = write_summary_report(results)

    train_total = results["train"]["summary"]["total_images"]
    val_total = results["val"]["summary"]["total_images"]
    test_total = results["test"]["summary"]["total_images"]
    combined_total = results["combined"]["summary"]["total_images"]
    overall_extension = results["combined"]["summary"]["most_common_extension"]
    overall_size = results["combined"]["summary"]["most_common_size"]
    unreadable_count = results["combined"]["summary"]["unreadable_count"]

    print("\nFinal Summary")
    print("=" * 40)
    print(f"Train total images: {train_total}")
    print(f"Val total images: {val_total}")
    print(f"Test total images: {test_total}")
    print(f"Combined total images: {combined_total}")
    print(f"Most common extension overall: {overall_extension}")
    print(f"Most common image size overall: {overall_size}")
    print(f"Number of unreadable images: {unreadable_count}")
    print(f"Reports folder path: {REPORTS_DIR}")
    print(f"Summary report path: {summary_report_path}")


if __name__ == "__main__":
    main()