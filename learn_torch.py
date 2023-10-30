import cv2
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.pth]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    images = np.array(images)

    dataset = TensorDataset(torch.tensor(images).float(), torch.tensor(labels).long())
    train_len = int((1 - TEST_SIZE) * len(dataset))
    test_len = len(dataset) - train_len
    train_data, test_data = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = TrafficNet().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    total_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * total_correct / len(test_data)}%")

    if len(sys.argv) == 3:
        torch.save(model.state_dict(), sys.argv[2])
        print(f"Model saved to {sys.argv[2]}.")

def load_data(data_dir):
    images = []
    labels = []
    for folder in range(NUM_CATEGORIES):
        files = os.listdir(os.path.join(data_dir, str(folder)))
        for file_name in files:
            image = cv2.imread(os.path.join(data_dir, str(folder), file_name))
            resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            images.append(resized_image.transpose(2, 0, 1) / 255.0)
            labels.append(folder)
    
    return (images, labels)

class TrafficNet(nn.Module):
    def __init__(self):
        super(TrafficNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, NUM_CATEGORIES)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 6)  # Adjusted reshaping
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    main()
