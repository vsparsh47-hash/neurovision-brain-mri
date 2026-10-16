# NeuroVision — Explainable Brain MRI Analysis

NeuroVision is an educational hackathon prototype for classifying brain MRI images into four dataset labels:
- Glioma
- Meningioma
- No tumor
- Pituitary tumor

It includes a Streamlit upload interface, a model-training script, evaluation metrics, and Grad-CAM explainability.

> **Medical disclaimer:** This is a research/educational prototype, not a clinically validated diagnostic tool. Never use its output to make medical decisions.

## 1. Dataset setup

Use a properly licensed public dataset, such as the Brain Tumor MRI Dataset on Kaggle:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Download and extract it locally. Arrange it like this (class folder names must match between train and test):

```text
data/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

The Kaggle dataset may use different capitalization or folder labels. Keep the original labels consistently, and inspect `train.py` output to confirm class order. Do not commit the dataset to GitHub.

## 2. Install

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train a model

Start with EfficientNetB0:

```bash
python train.py --data-dir data --model efficientnet_b0 --epochs 3
```

Other options:
```bash
python train.py --data-dir data --model resnet50 --epochs 3
python train.py --data-dir data --model custom_cnn --epochs 5
```

The script expects `data/Training` and `data/Testing`, each containing one folder per class. It saves a checkpoint under `models/` and class metadata in the checkpoint.

For a short hackathon, begin with 2–3 epochs to confirm the pipeline works. Training time depends on your hardware. GPU acceleration is recommended.

## 4. Evaluate

```bash
python evaluate.py --data-dir data --checkpoint models/efficientnet_b0.pth
```

Metrics and a confusion matrix are written to `outputs/`.

## 5. Run the app

```bash
streamlit run app.py
```

The app needs a trained checkpoint in `models/`. The class scores are model outputs, not calibrated clinical probabilities. Grad-CAM is an approximate visualization of model attention, not proof of a medically meaningful explanation.

## Project structure

```text
neurovision-brain-mri/
├── app.py
├── train.py
├── evaluate.py
├── gradcam.py
├── requirements.txt
├── .gitignore
├── data/             # local dataset; not committed
├── models/           # checkpoints; add only if repository size permits
└── outputs/          # generated metrics/plots
```

## Evaluation and limitations

- Report accuracy, macro F1, per-class precision/recall, and the confusion matrix.
- Avoid data leakage: keep all images from the same patient or source together when patient identifiers are available.
- Check for duplicate or near-duplicate images across splits.
- Performance on this dataset does not establish generalization to real clinical MRI scans or different hospitals/scanners.
- Include dataset attribution and license details in your submission.

## Team

Sparsh Verma , Abhinav Singh
## Experimental Results

All three models were evaluated on the same 1,600-image test set.

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| ResNet50 | 95.13% | 0.9506 |
| EfficientNetB0 | 94.50% | 0.9437 |
| Custom CNN | 58.06% | 0.5823 |

ResNet50 achieved the highest measured accuracy and macro F1-score among the three evaluated models.

### Explainability
Grad-CAM visualizations are provided for supported pretrained models to help inspect the image regions influencing predictions.

### Limitations
These results apply to the evaluated dataset and do not establish clinical validity or performance on MRI scans from other hospitals. The system is intended for research and educational use, not medical diagnosis.
