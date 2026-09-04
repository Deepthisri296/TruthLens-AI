const form = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const fileName = document.getElementById("fileName");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const statusMessage = document.getElementById("statusMessage");

const resultSection = document.getElementById("resultSection");
const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const resultMessage = document.getElementById("resultMessage");

const originalImage = document.getElementById("originalImage");
const heatmapImage = document.getElementById("heatmapImage");
const heatmapContainer = document.getElementById("heatmapContainer");

function displayFile(file) {
    if (!file) return;

    const validTypes = ["image/png", "image/jpeg"];

    if (!validTypes.includes(file.type)) {
        statusMessage.textContent = "Please select a PNG or JPG image.";
        statusMessage.className = "status-message error";
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        statusMessage.textContent = "File size must be less than 10 MB.";
        statusMessage.className = "status-message error";
        return;
    }

    fileName.textContent = file.name;
    statusMessage.textContent = "";
    statusMessage.className = "status-message";

    const reader = new FileReader();

    reader.onload = function(event) {
        previewImage.src = event.target.result;
        previewContainer.classList.remove("hidden");
    };

    reader.readAsDataURL(file);
}

imageInput.addEventListener("change", function() {
    displayFile(this.files[0]);
});

dropZone.addEventListener("dragover", function(event) {
    event.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", function(event) {
    event.preventDefault();
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", function(event) {
    event.preventDefault();
    dropZone.classList.remove("dragover");

    const file = event.dataTransfer.files[0];

    if (!file) return;

    imageInput.files = event.dataTransfer.files;
    displayFile(file);
});

form.addEventListener("submit", async function(event) {
    event.preventDefault();

    const file = imageInput.files[0];

    if (!file) {
        statusMessage.textContent = "Please select an image first.";
        statusMessage.className = "status-message error";
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    statusMessage.textContent = "Uploading and analyzing...";
    statusMessage.className = "status-message";

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Analysis failed.");
        }

        const result = data.result || {};

        let predictedLabel =
            result.label ||
            result.prediction ||
            result.status ||
            "UNKNOWN";

        predictedLabel = String(predictedLabel).toUpperCase();

        prediction.textContent = predictedLabel;

        let score = Number(result.confidence);

        if (Number.isFinite(score)) {
            if (score <= 1) {
                score = score * 100;
            }

            confidence.textContent =
                `Confidence: ${score.toFixed(2)}%`;
        } else {
            confidence.textContent = "Confidence: Not available";
        }

        if (result.message) {
            resultMessage.textContent = result.message;
        } else {
            resultMessage.textContent =
                `The model classified this image as ${predictedLabel}. ` +
                `The Grad-CAM heatmap shows the visual regions that influenced the prediction.`;
        }

        originalImage.src =
            `/testing-images/${encodeURIComponent(data.filename)}`;

        if (result.heatmap_url) {
            heatmapImage.src =
                result.heatmap_url + "?t=" + Date.now();

            heatmapContainer.classList.remove("hidden");
        } else {
            heatmapContainer.classList.add("hidden");
        }

        resultSection.classList.remove("hidden");

        statusMessage.textContent = "Analysis completed successfully.";
        statusMessage.className = "status-message success";

        setTimeout(function() {
            resultSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }, 300);

    } catch (error) {
        console.error(error);

        statusMessage.textContent = error.message;
        statusMessage.className = "status-message error";
    }
});