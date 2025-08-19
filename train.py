import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing and dataloader preparation
def prepare_data(dataset_path="./dataset", batch_size=32, val_split=0.25):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])    
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    num_classes = len(dataset.classes)

    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes, dataset.classes

# Simple CNN model
class CarClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*28*28, 512), nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x): return self.network(x)

# CNN model with batch normalization and dropout
class CarClassifierCNNWithRegularization(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*28*28, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x): return self.network(x)

# Transfer learning model using ResNet50
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Training loop with validation
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.to(device)
    start = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%")

    print(f"Training finished in {(time.time() - start):.2f}s")
    return all_labels, all_preds


# Model evaluation using classification report
def evaluate_model(labels, predictions):
    print("\nClassification Report:\n")
    print(classification_report(labels, predictions))

# Save trained model to file
def save_model(model, path="saved_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Main pipeline
def main():
    train_loader, val_loader, num_classes, class_names = prepare_data()

    # Select model
    # model = CarClassifierCNN(num_classes)
    # model = CarClassifierCNNWithRegularization(num_classes)
    model = CarClassifierResNet(num_classes, dropout_rate=0.2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.005
    )

    labels, predictions = train_model(model, train_loader, val_loader,
                                      criterion, optimizer, epochs=10)
    evaluate_model(labels, predictions)
    save_model(model, "saved_model.pth")


if __name__ == "__main__":
    main()


