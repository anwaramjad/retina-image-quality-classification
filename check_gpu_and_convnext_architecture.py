from pathlib import Path
import sys

import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


# هذا السكربت يتحقق من توفر GPU و CUDA.
# يقوم بتحميل نموذج ConvNeXt-Tiny وتعديل classifier ليخرج 3 فئات لجودة صور الشبكية.
# يتحقق من النموذج باستخدام input وهمي بحجم 512x512 وبثلاث قنوات RGB.
# ينشئ تقارير عن المعمارية والإعدادات ورسم توضيحي بسيط للنموذج.
# لا يقوم هذا السكربت بتدريب النموذج ولا يقرأ أو يعدل أي ملف من ملفات البيانات.


REPORTS_DIR = Path(r"C:\EyePACS\model_setup_reports")

FULL_ARCHITECTURE_PATH = REPORTS_DIR / "convnext_tiny_full_architecture.txt"
ARCHITECTURE_SUMMARY_CSV_PATH = REPORTS_DIR / "convnext_tiny_architecture_summary.csv"
TRAINING_CONFIGURATION_PATH = REPORTS_DIR / "training_configuration.txt"
ARCHITECTURE_DIAGRAM_PATH = REPORTS_DIR / "convnext_tiny_architecture_diagram.png"
MODEL_SETUP_REPORT_PATH = REPORTS_DIR / "model_setup_summary_report.txt"

MODEL_NAME = "ConvNeXt-Tiny"
TASK_NAME = "EyeQ Retinal Image Quality Classification"
NUM_CLASSES = 3
INPUT_SHAPE = (1, 3, 512, 512)
CLASS_NAMES = {
    0: "Good",
    1: "Usable",
    2: "Reject",
}


def create_reports_folder():
    # إنشاء مجلد التقارير فقط إذا لم يكن موجودا.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def bytes_to_gb(value):
    # تحويل الذاكرة من bytes إلى GB.
    return value / (1024 ** 3)


def get_environment_info():
    # جمع معلومات Python و PyTorch و CUDA والجهاز المستخدم.
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    info = {
        "python_version": sys.version.replace("\n", " "),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_available": torch.backends.cudnn.is_available(),
        "device_selected": str(device),
        "gpu_name": "",
        "gpu_total_memory_gb": "",
        "gpu_allocated_memory_before_model_gb": "",
        "gpu_reserved_memory_before_model_gb": "",
    }

    if cuda_available:
        gpu_index = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(gpu_index)

        info["gpu_name"] = torch.cuda.get_device_name(gpu_index)
        info["gpu_total_memory_gb"] = f"{bytes_to_gb(gpu_properties.total_memory):.2f}"
        info["gpu_allocated_memory_before_model_gb"] = f"{bytes_to_gb(torch.cuda.memory_allocated(gpu_index)):.4f}"
        info["gpu_reserved_memory_before_model_gb"] = f"{bytes_to_gb(torch.cuda.memory_reserved(gpu_index)):.4f}"

    return info, device


def print_environment_info(info):
    # طباعة معلومات البيئة بشكل واضح.
    print("Environment Check")
    print("=" * 40)
    print(f"Python version: {info['python_version']}")
    print(f"PyTorch version: {info['pytorch_version']}")
    print(f"Torchvision version: {info['torchvision_version']}")
    print(f"CUDA available: {info['cuda_available']}")
    print(f"CUDA version used by PyTorch: {info['torch_cuda_version']}")
    print(f"cuDNN available: {info['cudnn_available']}")
    print(f"Device selected: {info['device_selected']}")

    if info["cuda_available"]:
        print(f"GPU name: {info['gpu_name']}")
        print(f"GPU total memory GB: {info['gpu_total_memory_gb']}")
        print(f"GPU allocated memory before model loading GB: {info['gpu_allocated_memory_before_model_gb']}")
        print(f"GPU reserved memory before model loading GB: {info['gpu_reserved_memory_before_model_gb']}")
    else:
        print("GPU name: Not available")
        print("GPU total memory GB: Not available")


def load_convnext_model(device):
    # تحميل ConvNeXt-Tiny بأوزان ImageNet إن كانت متاحة ثم تعديل الطبقة الأخيرة إلى 3 فئات.
    print("\nLoading ConvNeXt-Tiny model")

    pretrained_status = "ImageNet pretrained weights"

    try:
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
        print("Loaded ImageNet pretrained weights")
    except Exception as error:
        weights = None
        pretrained_status = f"Pretrained weights were not loaded. Reason: {error}"
        model = convnext_tiny(weights=None)
        print(f"Warning: could not load pretrained weights. Using random weights. Reason: {error}")

    if not hasattr(model, "classifier"):
        raise AttributeError("Critical error: ConvNeXt model does not have a classifier attribute.")

    final_layer = model.classifier[-1]

    if not isinstance(final_layer, torch.nn.Linear):
        raise TypeError("Critical error: expected the final classifier layer to be torch.nn.Linear.")

    input_features = final_layer.in_features
    model.classifier[-1] = torch.nn.Linear(input_features, NUM_CLASSES)

    model = model.to(device)
    model.eval()

    print("Model classifier replaced with Linear layer for 3 classes")
    print(f"Final classifier layer: {model.classifier[-1]}")

    return model, pretrained_status


def verify_forward_pass(model, device):
    # تنفيذ forward pass وهمي بدون gradients للتأكد من شكل الخرج.
    print("\nRunning dummy forward pass")

    dummy_input = torch.randn(INPUT_SHAPE).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    output_shape = tuple(output.shape)
    expected_shape = (1, NUM_CLASSES)
    forward_pass_ok = output_shape == expected_shape

    print(f"Dummy input shape: {INPUT_SHAPE}")
    print(f"Model output shape: {list(output_shape)}")
    print(f"Expected output shape [1, 3]: {forward_pass_ok}")

    return output_shape, forward_pass_ok


def count_parameters(model):
    # حساب عدد معاملات النموذج الكلية والقابلة للتدريب وغير القابلة للتدريب.
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    non_trainable_parameters = total_parameters - trainable_parameters
    model_size_mb = (total_parameters * 4) / (1024 ** 2)

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "non_trainable_parameters": non_trainable_parameters,
        "model_size_mb": model_size_mb,
    }


def save_full_architecture(model):
    # حفظ النص الكامل لمعمارية النموذج.
    FULL_ARCHITECTURE_PATH.write_text(str(model), encoding="utf-8")
    print(f"Saved full architecture: {FULL_ARCHITECTURE_PATH}")


def save_architecture_summary_csv():
    # إنشاء جدول مختصر يشرح أقسام المعمارية.
    rows = [
        {
            "section_name": "Input",
            "description": "RGB retinal image tensor",
            "input_shape_if_known": "",
            "output_shape_if_known": "1 x 3 x 512 x 512",
            "notes": "Batch size 1, 3 RGB channels, image size 512x512",
        },
        {
            "section_name": "ConvNeXt feature extractor",
            "description": "Initial convolution stem and convolutional feature extraction layers",
            "input_shape_if_known": "1 x 3 x 512 x 512",
            "output_shape_if_known": "Feature maps",
            "notes": "Learns visual patterns from retinal images",
        },
        {
            "section_name": "ConvNeXt stages",
            "description": "Stacked ConvNeXt convolution blocks organized into stages",
            "input_shape_if_known": "Feature maps",
            "output_shape_if_known": "Deeper feature maps",
            "notes": "Architecture is described by stages, convolution blocks, channels, and feature maps",
        },
        {
            "section_name": "Global pooling / average pooling",
            "description": "Aggregates spatial feature maps before classification",
            "input_shape_if_known": "Deeper feature maps",
            "output_shape_if_known": "Compact feature vector",
            "notes": "Prepares extracted visual features for classifier head",
        },
        {
            "section_name": "Classifier head",
            "description": "Final classification module with a Linear layer",
            "input_shape_if_known": "Compact feature vector",
            "output_shape_if_known": "1 x 3",
            "notes": "Final Linear layer outputs 3 class logits",
        },
        {
            "section_name": "Final output",
            "description": "Raw logits for EyeQ quality classes",
            "input_shape_if_known": "1 x 3",
            "output_shape_if_known": "1 x 3",
            "notes": "Classes: 0 Good, 1 Usable, 2 Reject. No Softmax inside model during training",
        },
    ]

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(ARCHITECTURE_SUMMARY_CSV_PATH, index=False)
    print(f"Saved architecture summary CSV: {ARCHITECTURE_SUMMARY_CSV_PATH}")


def get_training_configuration_text():
    # تجهيز نص إعدادات التدريب المقترحة بدون تنفيذ أي تدريب.
    return "\n".join(
        [
            "Training Configuration",
            "=" * 40,
            "",
            "Task:",
            "3-Class Classification",
            "",
            "Classes:",
            "0 = Good",
            "1 = Usable",
            "2 = Reject",
            "",
            "Model:",
            "ConvNeXt-Tiny",
            "",
            "Pretrained:",
            "ImageNet pretrained weights",
            "",
            "Image Size:",
            "512x512",
            "",
            "Input Channels:",
            "3 RGB",
            "",
            "Final Layer:",
            "Linear layer with 3 outputs",
            "",
            "Activation:",
            "No Softmax inside model during training",
            "",
            "Loss Function:",
            "CrossEntropyLoss",
            "",
            "Imbalance Handling:",
            "Class Weights",
            "",
            "Optimizer:",
            "AdamW",
            "",
            "Learning Rate:",
            "3e-4",
            "",
            "Weight Decay:",
            "1e-4",
            "",
            "Batch Size:",
            "Start with 8 because image size is 512x512",
            "",
            "Epochs:",
            "25",
            "",
            "Early Stopping:",
            "Patience = 5",
            "",
            "Scheduler:",
            "ReduceLROnPlateau",
            "",
            "Main Validation Metric:",
            "Macro F1-score",
            "",
            "Other Metrics:",
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Per-class F1",
            "Confusion Matrix",
            "Classification Report",
            "",
            "Augmentation:",
            "Light augmentation only",
        ]
    )


def save_training_configuration():
    # حفظ ملف إعدادات التدريب المقترحة فقط بدون تدريب.
    training_text = get_training_configuration_text()
    TRAINING_CONFIGURATION_PATH.write_text(training_text, encoding="utf-8")
    print(f"Saved training configuration: {TRAINING_CONFIGURATION_PATH}")
    return training_text


def create_architecture_diagram():
    # إنشاء رسم بسيط لمسار البيانات داخل النموذج باستخدام matplotlib فقط.
    boxes = [
        ("Input Image\n512x512x3", 0.88),
        ("ConvNeXt-Tiny\nFeature Extractor", 0.70),
        ("Feature Maps /\nLearned Visual Features", 0.52),
        ("Global Pooling", 0.34),
        ("Classifier Head", 0.18),
        ("3 Outputs:\nGood\nUsable\nReject", 0.02),
    ]

    figure, axis = plt.subplots(figsize=(8, 10))
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.08, 1)
    axis.axis("off")

    box_width = 0.55
    box_height = 0.10
    x_center = 0.5

    for text, y_center in boxes:
        x_left = x_center - box_width / 2
        y_bottom = y_center - box_height / 2

        rectangle = plt.Rectangle(
            (x_left, y_bottom),
            box_width,
            box_height,
            fill=True,
            facecolor="#E3F2FD",
            edgecolor="#1565C0",
            linewidth=2,
        )
        axis.add_patch(rectangle)
        axis.text(
            x_center,
            y_center,
            text,
            ha="center",
            va="center",
            fontsize=12,
            color="#0D1B2A",
        )

    for index in range(len(boxes) - 1):
        start_y = boxes[index][1] - box_height / 2
        end_y = boxes[index + 1][1] + box_height / 2

        axis.annotate(
            "",
            xy=(x_center, end_y),
            xytext=(x_center, start_y),
            arrowprops={"arrowstyle": "->", "linewidth": 2, "color": "#333333"},
        )

    axis.set_title("ConvNeXt-Tiny Architecture for EyeQ 3-Class Classification", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(ARCHITECTURE_DIAGRAM_PATH, dpi=150)
    plt.close()

    print(f"Saved architecture diagram: {ARCHITECTURE_DIAGRAM_PATH}")


def format_environment_report(info):
    # تجهيز نص معلومات البيئة للتقرير النهائي.
    lines = [
        "Environment Check Results",
        "-" * 40,
        f"Python version: {info['python_version']}",
        f"PyTorch version: {info['pytorch_version']}",
        f"Torchvision version: {info['torchvision_version']}",
        f"CUDA available: {info['cuda_available']}",
        f"CUDA version used by PyTorch: {info['torch_cuda_version']}",
        f"cuDNN available: {info['cudnn_available']}",
        f"Device selected: {info['device_selected']}",
        f"GPU name: {info['gpu_name'] if info['gpu_name'] else 'Not available'}",
        f"GPU total memory in GB: {info['gpu_total_memory_gb'] if info['gpu_total_memory_gb'] else 'Not available'}",
        f"GPU allocated memory before model loading in GB: {info['gpu_allocated_memory_before_model_gb'] if info['gpu_allocated_memory_before_model_gb'] else 'Not available'}",
        f"GPU reserved memory before model loading in GB: {info['gpu_reserved_memory_before_model_gb'] if info['gpu_reserved_memory_before_model_gb'] else 'Not available'}",
    ]

    return "\n".join(lines)


def format_parameter_report(parameter_info):
    # تجهيز نص عدد المعاملات للتقرير النهائي.
    lines = [
        "Parameter Counts",
        "-" * 40,
        f"Total parameters: {parameter_info['total_parameters']}",
        f"Trainable parameters: {parameter_info['trainable_parameters']}",
        f"Non-trainable parameters: {parameter_info['non_trainable_parameters']}",
        f"Model size estimate in MB assuming float32 parameters: {parameter_info['model_size_mb']:.2f}",
    ]

    return "\n".join(lines)


def get_architecture_explanation(final_classifier_layer):
    # شرح مبسط لطريقة وصف ConvNeXt بدلا من مفهوم neurons per hidden layer.
    lines = [
        "Architecture Explanation",
        "-" * 40,
        "Model name: ConvNeXt-Tiny",
        "Task: 3-class classification",
        "Input shape: 1 x 3 x 512 x 512",
        "Output shape: 1 x 3",
        f"Final classifier layer: {final_classifier_layer}",
        f"Number of output classes: {NUM_CLASSES}",
        "",
        "ConvNeXt is a CNN architecture and does not use 'neurons per hidden layer' in the simple MLP sense.",
        "Instead, its structure is described using:",
        "stages",
        "convolution blocks",
        "channels",
        "feature maps",
        "classifier head",
    ]

    return "\n".join(lines)


def save_model_setup_summary_report(
    environment_info,
    pretrained_status,
    output_shape,
    forward_pass_ok,
    parameter_info,
    final_classifier_layer,
    training_configuration_text,
):
    # حفظ التقرير النهائي الذي يجمع البيئة والنموذج والإعدادات والملفات الناتجة.
    output_files = [
        FULL_ARCHITECTURE_PATH,
        ARCHITECTURE_SUMMARY_CSV_PATH,
        TRAINING_CONFIGURATION_PATH,
        ARCHITECTURE_DIAGRAM_PATH,
        MODEL_SETUP_REPORT_PATH,
    ]

    report_lines = [
        "ConvNeXt-Tiny Model Setup Summary Report",
        "=" * 50,
        "",
        format_environment_report(environment_info),
        "",
        "GPU Check Results",
        "-" * 40,
        f"CUDA is available: {environment_info['cuda_available']}",
        f"Selected device: {environment_info['device_selected']}",
        f"GPU name: {environment_info['gpu_name'] if environment_info['gpu_name'] else 'Not available'}",
        "",
        "Model Loading Status",
        "-" * 40,
        f"Model name: {MODEL_NAME}",
        f"Pretrained status: {pretrained_status}",
        "Classifier replacement: Completed",
        "Model mode: Evaluation mode for inspection",
        "",
        "Dummy Forward Pass Status",
        "-" * 40,
        f"Dummy input shape: {INPUT_SHAPE}",
        f"Output shape: {list(output_shape)}",
        f"Output shape is [1, 3]: {forward_pass_ok}",
        "",
        format_parameter_report(parameter_info),
        "",
        get_architecture_explanation(final_classifier_layer),
        "",
        training_configuration_text,
        "",
        "Output files created:",
    ]

    for output_file in output_files:
        report_lines.append(str(output_file))

    MODEL_SETUP_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved model setup summary report: {MODEL_SETUP_REPORT_PATH}")


def main():
    # تشغيل خطوات فحص البيئة وتحميل النموذج وإنشاء التقارير.
    print("Starting GPU and ConvNeXt-Tiny architecture setup check")
    create_reports_folder()

    environment_info, device = get_environment_info()
    print_environment_info(environment_info)

    model, pretrained_status = load_convnext_model(device)
    output_shape, forward_pass_ok = verify_forward_pass(model, device)

    parameter_info = count_parameters(model)
    final_classifier_layer = model.classifier[-1]

    print("\nParameter Counts")
    print("=" * 40)
    print(f"Total parameters: {parameter_info['total_parameters']}")
    print(f"Trainable parameters: {parameter_info['trainable_parameters']}")
    print(f"Non-trainable parameters: {parameter_info['non_trainable_parameters']}")
    print(f"Model size estimate MB: {parameter_info['model_size_mb']:.2f}")

    save_full_architecture(model)
    save_architecture_summary_csv()
    training_configuration_text = save_training_configuration()
    create_architecture_diagram()

    save_model_setup_summary_report(
        environment_info=environment_info,
        pretrained_status=pretrained_status,
        output_shape=output_shape,
        forward_pass_ok=forward_pass_ok,
        parameter_info=parameter_info,
        final_classifier_layer=final_classifier_layer,
        training_configuration_text=training_configuration_text,
    )

    print("\nFinal Summary")
    print("=" * 40)
    print(f"Device used: {environment_info['device_selected']}")
    print(f"CUDA available: {environment_info['cuda_available']}")
    print(f"GPU name: {environment_info['gpu_name'] if environment_info['gpu_name'] else 'Not available'}")
    print(f"Model name: {MODEL_NAME}")
    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Output shape: {list(output_shape)}")
    print(f"Total parameters: {parameter_info['total_parameters']}")
    print(f"Trainable parameters: {parameter_info['trainable_parameters']}")
    print(f"Reports folder path: {REPORTS_DIR}")
    print("Confirmation that no training was performed: Yes")


if __name__ == "__main__":
    main()
