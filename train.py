"""Train a transfer-learning flower classifier on ImageFolder datasets."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ARCHITECTURES = {"vgg16", "densenet121"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(architecture="vgg16", hidden_units=512, pretrained=True):
    """Freeze a pretrained backbone and replace its classifier."""
    if architecture == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
    elif architecture == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    input_features = (model.classifier[0].in_features
                      if architecture == "vgg16" else model.classifier.in_features)
    model.classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1),
    )
    return model


def make_loaders(data_dir, batch_size=64):
    """Expect train/, valid/ and test/ directories of class-labelled images."""
    root = Path(data_dir)
    augmentation = transforms.Compose([
        transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    evaluation = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    datasets_by_split = {
        split: datasets.ImageFolder(root / split, transform=augmentation if split == "train" else evaluation)
        for split in ("train", "valid", "test")
    }
    classes = datasets_by_split["train"].class_to_idx
    if len(classes) != 102:
        raise ValueError(f"Expected 102 flower classes, found {len(classes)}")
    for split in ("valid", "test"):
        if datasets_by_split[split].class_to_idx != classes:
            raise ValueError(f"Class mapping differs for {split} split")
    loaders = {
        split: DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
        for split, ds in datasets_by_split.items()
    }
    return loaders, classes


def evaluate(model, loader, device):
    """Compute sample-weighted negative log-likelihood and accuracy."""
    model.eval()
    loss_total, correct, count = 0.0, 0, 0
    criterion = nn.NLLLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_total += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return {"loss": loss_total / count, "accuracy": correct / count}


def train(model, loaders, device, epochs, learning_rate):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total, count = 0.0, 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * labels.size(0)
            count += labels.size(0)
        validation = evaluate(model, loaders["valid"], device)
        record = {"epoch": epoch, "train_loss": loss_total / count,
                  "valid_loss": validation["loss"], "valid_accuracy": validation["accuracy"]}
        history.append(record)
        print(f"Epoch {epoch}/{epochs}: train loss={record['train_loss']:.4f}, "
              f"validation loss={record['valid_loss']:.4f}, "
              f"validation accuracy={record['valid_accuracy']:.2%}")
    return optimizer, history


def save_checkpoint(path, model, architecture, hidden_units, class_to_idx, epochs, optimizer):
    """Store only primitive metadata and tensor state dictionaries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": architecture, "hidden_units": hidden_units,
        "class_to_idx": class_to_idx, "epochs": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Folder containing train/, valid/ and test/")
    parser.add_argument("--save_dir", default="checkpoints/model.pth", help="Checkpoint file path")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="vgg16")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_units", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true", help="Use CUDA if available")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.hidden_units < 1 or args.batch_size < 1:
        raise ValueError("epochs, hidden_units and batch_size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    loaders, class_to_idx = make_loaders(args.data_dir, args.batch_size)
    model = build_model(args.arch, args.hidden_units).to(device)
    optimizer, history = train(model, loaders, device, args.epochs, args.learning_rate)
    test = evaluate(model, loaders["test"], device)
    print(f"Test accuracy: {test['accuracy']:.2%}; test loss: {test['loss']:.4f}")
    save_checkpoint(args.save_dir, model, args.arch, args.hidden_units,
                    class_to_idx, args.epochs, optimizer)
    metrics_path = Path(args.save_dir).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"history": history, "test": test}, indent=2))
    print(f"Saved checkpoint: {args.save_dir}")


if __name__ == "__main__":
    main()
