# TruthLens AI

TruthLens AI is an AI-powered image verification system that classifies images as **Real** or **Fake** using a deep learning model.

## Features

- Real/Fake image classification
- Confidence score
- Grad-CAM heatmap visualization
- Explainable AI (XAI) explanation
- Flask-based web interface
- Image upload with preview

## Tech Stack

- Python
- PyTorch
- ResNet + CBAM
- Grad-CAM
- OpenCV
- Flask
- HTML, CSS, JavaScript

## Run the Project

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
Start the Flask application:

python app\app.py

Open:

http://127.0.0.1:5000
Output

The system provides:

Prediction → Confidence → Grad-CAM Heatmap → XAI Explanation

Model Performance
Accuracy: 94.00%
Precision: 95.08%
Recall: 92.80%
F1 Score: 93.93%