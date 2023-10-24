import cv2
import csv
import numpy as np
import os
import sys
import tensorflow as tf
import learn


def main():
    # Check command-line arguments
    if len(sys.argv) != 3:
        sys.exit("Usage: python predict.py model image_directory")

    # Load sign dictionary
    with open("gtsrb/sign_descriptions.csv", "r", encoding="utf-8-sig") as file:
        signs = list(csv.DictReader(file))

    # Load images
    images, filenames = load_data(sys.argv[2])

    # Load existing tf model from file
    model = tf.keras.models.load_model(sys.argv[1])

    # Get predictions from model
    predictions = model.predict(np.array(images))

    # Add most-likely prediction and confidence level to each image dictionary
    print_predictions(filenames, predictions, signs)


def load_data(directory):
    """Read images into list and resize to match training images"""

    images = []
    filenames = []

    for file in sorted(os.listdir(directory)):
        image = cv2.imread(os.path.join(directory, file))
        resized_image = cv2.resize(image, (learn.IMG_WIDTH, learn.IMG_HEIGHT))
        images.append(resized_image / 255.0)
        filenames.append(file)

    return images, filenames


def print_predictions(filenames, predictions, signs):
    """Prints prediction"""

    print()
    for file, prediction in enumerate(predictions):
        print(
            f"{filenames[file]} is {signs[np.argmax(prediction)]['description']} sign with a confidence of {100 * np.max(prediction):.2f}%"
        )


if __name__ == "__main__":
    main()
