# 🧠 NeuroVision: Explainable Brain MRI Classification

**ML × ACS Hackathon 2026 | Medical Imaging & Explainable Artificial Intelligence**

Python 3.10+ · PyTorch · TorchVision · Streamlit · Grad-CAM · License: MIT

NeuroVision is an interactive, machine learning-based brain MRI classification research and educational prototype. It is designed to classify uploaded brain MRI images into four categories—**Glioma, Meningioma, Pituitary Tumor, and No Tumor**—and support model interpretability through explainability visualizations.

The project combines deep learning, medical image classification, and explainable AI in an interactive web application built with Streamlit.

> ⚠️ **Research and educational prototype only. NeuroVision is not a medical diagnostic tool. Its predictions must not be used for clinical decisions.**

## 📌 Executive Summary

NeuroVision explores the application of convolutional neural networks (CNNs) and transfer learning to brain MRI image classification.

The project includes three model architectures:

- **ResNet50:** A residual convolutional neural network used as a deep learning classification model.
- **EfficientNet-B0:** A computationally efficient CNN architecture for image classification.
- **Custom CNN:** A custom-designed convolutional neural network trained for the four-class classification task.

The project stores model checkpoints separately to support model loading, evaluation, comparison, and future integration into the web application.

**Important:** The existence of a checkpoint file does not independently establish that training completed correctly or that the model is accurate. Model performance should be reported only after reproducible evaluation on held-out data.

### Classification Categories

| Class | Description |
|---|---|
| Glioma | MRI image labelled as a glioma case |
| Meningioma | MRI image labelled as a meningioma case |
| No Tumor | MRI image labelled as containing no tumor |
| Pituitary Tumor | MRI image labelled as a pituitary tumor case |

The output represents a model prediction for an image, not a confirmed medical diagnosis.

## 🎯 Project Objectives

- Develop an interactive brain MRI classification application.
- Explore transfer learning with ResNet50 and EfficientNet-B0.
- Implement a custom CNN for comparison with established architectures.
- Investigate explainable AI methods such as Grad-CAM.
- Provide an accessible interface for uploading MRI images and viewing model outputs.
- Establish a reproducible workflow for training, evaluation, and deployment.

## 🎨 Application Highlights

### 1. 🧠 Brain MRI Classification

Users can upload an MRI image through the Streamlit interface and obtain a prediction from a supported model.

### 2. 🔬 Multiple Deep Learning Architectures

NeuroVision includes three model checkpoints:

- `models/resnet50.pth`
- `models/efficientnet_b0.pth`
- `models/custom_cnn.pth`

Each checkpoint must be loaded using its corresponding architecture and compatible preprocessing pipeline.

### 3. 🔍 Explainable AI with Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) can help visualize image regions that influence a CNN's prediction. These visualizations are intended for research and model interpretation; they do not establish that a prediction is clinically correct or that highlighted regions represent tumor boundaries.

### 4. 🌐 Interactive Streamlit Interface

The web application provides an accessible interface for image uploads and model inference, with scope for model selection, prediction visualization, and explainability features.

### 5. 📊 Model Evaluation

The project can be evaluated using appropriate classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class performance

No performance figures are reported here because verified evaluation results have not yet been documented.

## 🏗️ Model Architecture

### ResNet50

ResNet50 uses residual connections to facilitate the training of deep convolutional neural networks. It provides a baseline architecture for evaluating brain MRI classification.

**Checkpoint:** `models/resnet50.pth`

### EfficientNet-B0

EfficientNet-B0 uses a compound scaling approach to balance network depth, width, and input resolution. It provides an alternative architecture for investigating the trade-off between predictive performance and computational cost.

**Checkpoint:** `models/efficientnet_b0.pth`

### Custom CNN

The Custom CNN uses convolutional layers to extract image features, followed by classification layers that map learned features to the four target classes.

**Checkpoint:** `models/custom_cnn.pth`

The exact layer configuration should be documented alongside the implementation in the training code.

## 🗂️ Project Architecture

```text
neurovision-brain-mri/
├── app.py
│   └── Main Streamlit web application
├── train.py
│   └── Model training pipeline
├── requirements.txt
│   └── Python dependencies
├── README.md
│   └── Project documentation
├── models/
│   ├── resnet50.pth
│   ├── efficientnet_b0.pth
│   └── custom_cnn.pth
└── data/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

*Note: This tree describes the expected layout. Adjust it to match the actual files and folders in the repository.*

## 🚀 Setup & Local Installation

### Prerequisites

- Python 3.10 or a compatible Python version supported by your dependencies.
- Git.
- VS Code or another Python development environment.
- Sufficient RAM and storage for PyTorch and the MRI dataset.

### 1. Clone the Repository

```bash
git clone https://github.com/vsparsh47-hash/neurovision-brain-mri.git
cd neurovision-brain-mri
```

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the requirements file does not include the required packages, install the missing dependencies explicitly:

```bash
pip install torch torchvision streamlit scikit-learn matplotlib pillow
```

The final dependency list should match the imports used by the application and training pipeline.

## 💻 Running the Application

Start the Streamlit application locally:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

**http://localhost:8501**

To specify a port:

```bash
python -m streamlit run app.py --server.port 8501
```

## 🏋️ Model Training

The repository includes a training script named `train.py`.

Ensure that the MRI dataset is present and organized into the expected class folders before training.

### Train ResNet50

```bash
python train.py --data-dir data --model resnet50 --epochs 3
```

### Train EfficientNet-B0 and Custom CNN

The exact model identifiers and supported arguments depend on the implementation of `train.py`. Use the model names accepted by that script and confirm that each training run saves its checkpoint to the expected path.

Expected checkpoint filenames:

```text
models/resnet50.pth
models/efficientnet_b0.pth
models/custom_cnn.pth
```

The three-epoch command is a preliminary training run, not a guarantee of convergence or acceptable performance. Training duration, hyperparameters, and model selection should be determined using validation data.

## 🧪 Testing & Verification

Before deployment, verify the following.

### Checkpoint Verification

- Confirm that all expected checkpoint files exist.
- Confirm that each checkpoint loads successfully with its matching architecture.
- Confirm that the number and ordering of output classes match the application's class mapping.
- Confirm that inference preprocessing matches the preprocessing used during training.

### Prediction Verification

- Test valid MRI image uploads.
- Test unsupported image formats and corrupted files.
- Confirm that the application handles missing checkpoints gracefully.
- Verify that predictions and confidence scores are displayed correctly.
- Confirm that Grad-CAM works with the selected model, where supported.

### Model Evaluation

Evaluate each model on a held-out test dataset that was not used for training or model selection. Report per-class precision, recall, F1-score, confusion matrices, and overall accuracy only after the evaluation has been executed and verified.

For a fair comparison, use the same test split and consistent evaluation procedures across all three architectures.

## ☁️ Deployment on Streamlit Community Cloud

NeuroVision can be deployed using Streamlit Community Cloud.

1. Push the application code to GitHub.
2. Ensure that the required dependencies are listed in `requirements.txt`.
3. Confirm that the deployed application can access all required model checkpoints.
4. Open [Streamlit Community Cloud](https://share.streamlit.io/).
5. Connect the GitHub repository and select the correct branch and `app.py` entry point.
6. Deploy the application and inspect the logs if an error occurs.

### Deployment Considerations

- GitHub rejects individual files larger than 100 MB. The ResNet50 checkpoint should be checked against this limit before pushing.
- Checkpoints excluded by `.gitignore` will not be uploaded unless explicitly added.
- Large models may require Git LFS or external model storage.
- Streamlit Community Cloud has resource limits; model loading and inference must be tested within those limits.
- Never commit credentials, access tokens, or other secrets to the repository.

## 🔐 Limitations & Responsible Use

NeuroVision is a research and educational prototype. Its output can be affected by image quality, dataset composition, class imbalance, scanner differences, preprocessing, and model generalization.

A model prediction or Grad-CAM visualization should not be interpreted as a definitive medical finding.

The application is not validated for clinical diagnosis, treatment selection, or patient management. Any medical concern should be assessed by a qualified healthcare professional using appropriate clinical evaluation and validated diagnostic procedures.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning and model inference |
| TorchVision | CNN architectures and image transformations |
| Streamlit | Interactive web application |
| Pillow | Image loading and processing |
| Matplotlib | Visualization support |
| Grad-CAM | Model explainability, where implemented |
| Git & GitHub | Version control and collaboration |
| Streamlit Community Cloud | Web application hosting |

## 🔮 Future Enhancements

- Interactive model selection between ResNet50, EfficientNet-B0, and Custom CNN.
- Side-by-side model performance comparison.
- Improved Grad-CAM visualization and explanation controls.
- Automated testing for checkpoint loading and inference.
- Validation-based hyperparameter tuning.
- Improved model artifact management and deployment reliability.
- Reproducible training and evaluation reports.

## 👥 Contributors & Acknowledgements

**Project:** NeuroVision — Explainable Brain MRI Classification

**Repository:** [vsparsh47-hash/neurovision-brain-mri](https://github.com/vsparsh47-hash/neurovision-brain-mri)

Developed as a machine learning and medical imaging research and educational project.

## 📄 License

This project is intended to be distributed under the MIT License. Ensure that a `LICENSE` file is present in the repository before claiming that the project is formally licensed under MIT.

---

**Disclaimer:** NeuroVision is not a medical diagnostic tool. Its predictions are intended for research and educational exploration only and must not be used for clinical decisions.
