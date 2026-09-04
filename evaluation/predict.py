import sys
from pathlib import Path

import torch
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms

from models.resnet_cbam import ResNetCBAM
from attention.gradcam import GradCAM
from attention.xai import generate_explanation


BASE_DIR = Path(__file__).resolve().parent.parent

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = BASE_DIR / "best_model.pth"

CLASSES = ["fake", "real"]

RESULTS_DIR = BASE_DIR / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


print("Loading TruthLens-AI model...")

model = ResNetCBAM(num_classes=2).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully!")
print("Device:", DEVICE)


target_layer = model.features[7][2].conv2

gradcam = GradCAM(
    model=model,
    target_layer=target_layer
)


def predict_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print("\nProcessing image:")
    print(image_path)

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    cam, output = gradcam.generate(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    prediction = output.argmax(dim=1).item()

    confidence = (
        probabilities[0][prediction].item() * 100
    )

    predicted_class = CLASSES[prediction]

    explanation = generate_explanation(
        predicted_class,
        confidence
    )

    heatmap = cam[0].detach().cpu().numpy()

    heatmap = cv2.resize(
        heatmap,
        (image.width, image.height)
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    original = np.array(image)

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    heatmap_filename = f"gradcam_{image_path.stem}.jpg"

    heatmap_path = RESULTS_DIR / heatmap_filename

    Image.fromarray(overlay).save(heatmap_path)

    print("\n==========================================")
    print("           TRUTHLENS-AI RESULT")
    print("==========================================")
    print(
        "Prediction  :",
        predicted_class.upper()
    )
    print(
        f"Confidence  : {confidence:.2f}%"
    )
    print("==========================================")

    print("\n==========================================")
    print("             XAI EXPLANATION")
    print("==========================================")
    print(explanation)
    print("==========================================")

    print("\nGrad-CAM saved to:", heatmap_path)

    print(
        "\nTruthLens-AI analysis completed successfully!"
    )

    return {
        "prediction": predicted_class.upper(),
        "label": predicted_class.upper(),
        "confidence": round(confidence, 2),
        "message": explanation,
        "explanation": explanation,
        "heatmap_filename": heatmap_filename,
        "heatmap_path": str(heatmap_path)
    }


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print(
            "python -m evaluation.predict <image_path>"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    result = predict_image(image_path)

    print("\nReturned result:")
    print(result)