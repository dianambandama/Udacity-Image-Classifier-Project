import argparse
import json
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F


def parse_arguments():
    """Parse command-line arguments for image prediction."""
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model.")
    parser.add_argument('image_path', type=str, default='flowers/test/58/image_02663.jpg',
                        help="Path to the input image.")
    parser.add_argument('checkpoint', type=str, default='train_checkpoint.pth',
                        help="Path to the model checkpoint file.")
    parser.add_argument('--top_k', type=int, default=5,
                        help="Number of top predictions to return.")
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help="Path to the JSON file mapping categories to names.")
    parser.add_argument('--gpu', action='store_true', default=False,
                        help="Use GPU for inference if available.")
    return parser.parse_args()


def load_model_from_checkpoint(filepath):
    """Load a model from a checkpoint file."""
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = getattr(torchvision.models, checkpoint['pretrained_model'])(pretrained=True)
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']
    model.learning_rate = checkpoint['learning_rate']
    model.hidden_units = checkpoint['hidden_units']
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']
    return model


def preprocess_image(image):
    """Preprocess an image for model input."""
    resize_size = 256
    crop_size = 224
    width, height = image.size

    # Resize the image while maintaining aspect ratio
    if height > width:
        height = int(max(height * resize_size / width, 1))
        width = resize_size
    else:
        width = int(max(width * resize_size / height, 1))
        height = resize_size

    image = image.resize((width, height))

    # Crop the center of the image
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))

    # Normalize the image
    image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))  # Rearrange dimensions to (C, H, W)
    return image


def predict_image_class(image_path, model, top_k, use_gpu):
    """Predict the top K classes for an image."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path)
    image = preprocess_image(image)
    image = torch.from_numpy(image).unsqueeze(0).float()

    # Perform inference
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        probabilities = F.softmax(output, dim=1)

    # Get the top K probabilities and classes
    top_probabilities, top_indices = probabilities.topk(top_k)
    top_probabilities = top_probabilities.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    # Map indices to class labels
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [index_to_class[index] for index in top_indices]

    return top_probabilities, top_classes, device


def load_category_names(category_names_file):
    """Load category names from a JSON file."""
    with open(category_names_file, 'r') as file:
        category_names = json.load(file)
    return category_names


def main():
    """Main function to predict the class of an image."""
    args = parse_arguments()
    image_path = args.image_path
    checkpoint_path = args.checkpoint
    top_k = args.top_k
    category_names_file = args.category_names
    use_gpu = args.gpu

    # Load the model from the checkpoint
    model = load_model_from_checkpoint(checkpoint_path)

    # Predict the top K classes for the image
    probabilities, classes, device = predict_image_class(image_path, model, top_k, use_gpu)

    # Load category names
    category_names = load_category_names(category_names_file)
    labels = [category_names[str(cls)] for cls in classes]

    # Print the results
    print(f"Results for your file: {image_path}")
    print(f"Top {top_k} predicted classes: {labels}")
    print(f"Probabilities: {probabilities}")
    print()

    for i, (label, probability) in enumerate(zip(labels, probabilities), start=1):
        print(f"{i} - {label} with a probability of {probability:.4f}")


if __name__ == "__main__":
    main()