import argparse
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from train import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="models/efficientnet_b0.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}. Train a model first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt["classes"]
    size = ckpt.get("image_size", 224)
    transform = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(Path(args.data_dir) / "Testing", transform=transform)
    if test_ds.classes != classes:
        raise SystemExit(f"Checkpoint classes {classes} do not match test folders {test_ds.classes}")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(ckpt["model_name"], len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    summary = {
        "model": ckpt["model_name"],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist(),
        "classes": classes,
    }
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out / f"{ckpt['model_name']}_metrics.json").write_text(json.dumps(summary, indent=2))
    cm = np.array(summary["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix — {ckpt['model_name']}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / f"{ckpt['model_name']}_confusion_matrix.png", dpi=160)
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Macro F1: {summary['macro_f1']:.4f}")
    print(f"Saved metrics and confusion matrix to {out}/")


if __name__ == "__main__":
    main()
