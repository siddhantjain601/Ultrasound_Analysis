import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
#DATA_DIR = "data"
NUM_CLASSES = 3
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3

# Data transforms
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# # Dataset and loaders
# dataset = ImageFolder(DATA_DIR, transform=transform)
# class_names = dataset.classes
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_ds, val_ds = random_split(dataset, [train_size, val_size])
# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ----------------------------
# ResNet Building Blocks
# ----------------------------

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

# ResNet18
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers
        self.layer1 = self.make_layer(block, 64,  layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# ----------------------------
# Training and Evaluation
# ----------------------------

# model = resnet18(num_classes=NUM_CLASSES).to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# criterion = nn.CrossEntropyLoss()

# def train():
#     model.train()
#     for epoch in range(EPOCHS):
#         total_loss = 0
#         correct = 0
#         for images, labels in train_loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             correct += (outputs.argmax(1) == labels).sum().item()

#         acc = correct / len(train_ds)
#         print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# def evaluate():
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             outputs = model(images)
#             preds = outputs.argmax(1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#     print(classification_report(all_labels, all_preds, target_names=class_names))

