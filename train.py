import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import json


def parse_arguments():
    """Parse command-line arguments for training the model."""
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset.")
    parser.add_argument('data_dir', type=str, help="Directory containing the dataset.")
    parser.add_argument('--save_dir', type=str, default='train_checkpoint.pth',
                        help="Directory to save the model checkpoint.")
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'],
                        help="Model architecture to use (vgg16 or densenet121).")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--hidden_units', type=int, default=512,
                        help="Number of hidden units in the classifier.")
    parser.add_argument('--epochs', type=int, default=1,
                        help="Number of epochs to train the model.")
    parser.add_argument('--gpu', action='store_true', default=False,
                        help="Use GPU for training if available.")
    return parser.parse_args()


def initialize_model(architecture, hidden_units, use_gpu):
    """Initialize a pre-trained model and modify its classifier."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    if architecture == "vgg16":
        model = models.vgg16(pretrained=True)
        num_input_features = 25088
    else:
        model = models.densenet121(pretrained=True)
        num_input_features = 1024

    # Freeze model parameters to avoid backpropagation through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_input_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.to(device)
    return model, device, num_input_features


def train_model(epochs, train_loader, validation_loader, model, device, criterion, optimizer):
    """Train the model and validate it periodically."""
    steps = 0
    running_loss = 0
    print_every = 5

    start_time = time.time()
    print("Training the model...")

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        loss = criterion(logps, labels)
                        validation_loss += loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Training Loss: {running_loss / print_every:.3f}.. "
                      f"Validation Loss: {validation_loss / len(validation_loader):.3f}.. "
                      f"Validation Accuracy: {accuracy / len(validation_loader):.3f}")
                running_loss = 0
                model.train()

    total_time = time.time() - start_time
    print(f"Model trained in: {total_time // 60:.0f}m {total_time % 60:.0f}s")


def save_checkpoint(file_path, model, datasets, epochs, optimizer, learning_rate, input_size, output_size, architecture, hidden_units):
    """Save the model checkpoint to a file."""
    model.class_to_idx = datasets[0].class_to_idx
    checkpoint = {
        'architecture': architecture,
        'input_size': input_size,
        'output_size': output_size,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'classifier': model.classifier,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state': optimizer.state_dict()
    }

    torch.save(checkpoint, file_path)
    print(f"Model checkpoint saved to {file_path}.")


def main():
    """Main function to train the model."""
    print("Loading data...")
    args = parse_arguments()
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'

    # Define transformations for the training, validation, and testing datasets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    datasets = [
        datasets.ImageFolder(train_dir, transform=train_transforms),
        datasets.ImageFolder(valid_dir, transform=valid_transforms),
        datasets.ImageFolder(test_dir, transform=test_transforms)
    ]

    # Create data loaders
    dataloaders = [
        torch.utils.data.DataLoader(datasets[0], batch_size=64, shuffle=True),
        torch.utils.data.DataLoader(datasets[1], batch_size=64, shuffle=True),
        torch.utils.data.DataLoader(datasets[2], batch_size=64, shuffle=True)
    ]

    # Load category-to-name mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Initialize the model, device, and input features
    model, device, num_input_features = initialize_model(args.arch, args.hidden_units, args.gpu)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train the model
    train_model(args.epochs, dataloaders[0], dataloaders[1], model, device, criterion, optimizer)

    # Save the model checkpoint
    save_checkpoint(args.save_dir, model, datasets, args.epochs, optimizer, args.learning_rate,
                    num_input_features, 102, args.arch, args.hidden_units)


if __name__ == "__main__":
    main()