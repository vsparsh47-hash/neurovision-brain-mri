import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def build_model(name: str, num_classes: int):
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "custom_cnn":
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), nn.Linear(128, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", choices=["custom_cnn", "resnet50", "efficientnet_b0"], default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir, test_dir = data_dir / "Training", data_dir / "Testing"
    if not train_dir.exists() or not test_dir.exists():
        raise SystemExit("Expected data/Training and data/Testing folders. See README.md.")

    train_tfms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)
    if train_ds.classes != test_ds.classes:
        raise SystemExit(f"Class folders differ. Training={train_ds.classes}; Testing={test_ds.classes}")
    if len(train_ds.classes) < 2:
        raise SystemExit("Need at least two class folders.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes (output order): {train_ds.classes}")
    print(f"Images: train={len(train_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(args.model, len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}/{args.epochs} | loss={running_loss/max(total,1):.4f} | train_acc={correct/max(total,1):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels.to(device)).sum().item()
            total += labels.size(0)
    print(f"Holdout accuracy: {correct/max(total,1):.4f}")

    out = Path("models")
    out.mkdir(exist_ok=True)
    checkpoint = out / f"{args.model}.pth"
    torch.save({
        "model_name": args.model,
        "state_dict": model.cpu().state_dict(),
        "classes": train_ds.classes,
        "image_size": args.image_size,
    }, checkpoint)
    print(f"Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
