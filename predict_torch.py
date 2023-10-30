import cv2
import csv
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import learn_torch


def main():
    # Check command-line arguments
    if len(sys.argv) != 3:
        sys.exit("Usage: python predict_torch.py model_path image_directory")

    # Ensure you're using the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load sign dictionary
    with open("gtsrb/sign_descriptions.csv", "r", encoding="utf-8-sig") as file:
        signs = list(csv.DictReader(file))

    # Load images
    images, filenames = load_data(sys.argv[2])

    images = np.array(images)  # Convert list of numpy arrays to a single numpy array
    images_tensor = torch.tensor(images).float().to(device)

    # Load the PyTorch model
    model = learn_torch.TrafficNet().to(device)
    model.load_state_dict(torch.load(sys.argv[1]))
    model = model.to(device)
    model.eval()

    # Get predictions from model
    with torch.no_grad():
        outputs = model(images_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    # Print predictions
    print_predictions(filenames, outputs.cpu(), signs)


def load_data(directory):
    """Read images into list and resize to match training images"""

    images = []
    filenames = []

    for file in sorted(os.listdir(directory)):
        image = cv2.imread(os.path.join(directory, file))
        resized_image = cv2.resize(image, (learn_torch.IMG_WIDTH, learn_torch.IMG_HEIGHT))
        images.append(resized_image.transpose((2, 0, 1)) / 255.0)  # Transpose to CxHxW format for PyTorch
        filenames.append(file)

    return images, filenames


def print_predictions(filenames, predictions, signs):
    """Prints prediction"""
    
    # If you need the numpy version of predictions:
    # predictions_np = predictions.numpy()
    
    print()
    for file, prediction in enumerate(predictions):
        probabilities = F.softmax(prediction, dim=0)
        confidence, predicted = probabilities.max(0)

        print(
            f"{filenames[file]} is {signs[predicted.item()]['description']} sign with a confidence of {confidence.item():.2%}"
        )


if __name__ == "__main__":
    main()
