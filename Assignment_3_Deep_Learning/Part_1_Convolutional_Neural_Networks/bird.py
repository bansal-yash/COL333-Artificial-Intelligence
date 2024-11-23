import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import csv

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch.cuda.empty_cache()


class sq_and_exp_block(nn.Module):

    def __init__(self, in_channels, reduction=16):

        super(sq_and_exp_block, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # Squeeze
        y = self.squeeze(x).view(batch_size, channels)

        # Excitation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)

        # Recalibrate
        return x * y.expand_as(x)


class conv_block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(conv_block, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = sq_and_exp_block(out_channels, reduction=16)

        # Downsampling for matching dimensions in skip connections
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class col333_model(nn.Module):
    def __init__(self, num_classes=10):
 
        super(col333_model, self).__init__()

        # Initial main layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 base layers of conv block
        self.layer1 = self.base_layer(conv_block, 64, 128, 2)
        self.layer2 = self.base_layer(conv_block, 128, 256, 2, stride=2)
        self.layer3 = self.base_layer(conv_block, 256, 512, 2, stride=2)

        # Classifier layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * conv_block.expansion, num_classes)

        # Weight initialisation
        self._initialize_weights()

    def base_layer(self, block, in_channels, out_channels, blocks, stride=1):

        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:

            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))

        for _ in range(1, blocks):
            layers.append(block(out_channels * block.expansion, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        # Intial conv layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 4 base layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Classifier layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def calculate_class_weights(train_dataset: datasets.ImageFolder):
    dataset = train_dataset

    class_counts = torch.tensor(
        [dataset.targets.count(i) for i in range(len(dataset.classes))],
        dtype=torch.float,
    )
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()

    return class_weights.to(device)


def train_model(
    model: col333_model,
    train_loader: DataLoader,
    criterion,
    optimizer,
    model_path,
    num_epochs=80,
):

    train_losses = []

    print("Training started")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.5f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]}")

        scheduler.step()

        torch.save(model.state_dict(), model_path)

    torch.save(model.state_dict(), model_path)
    print("Training ended")


def test_model(model: col333_model, test_loader: DataLoader, output_csv="bird.csv"):
    model.eval()
    results = []

    print("Testing started")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            results.extend(predicted.cpu().numpy())

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Predicted_Label"])
        for label in results:
            writer.writerow([label])

    print("Testing ended")


dataPath = sys.argv[1]
trainStatus = sys.argv[2]
modelPath = sys.argv[3] if len(sys.argv) > 3 else "bird.pth"
root_dir = dataPath

if trainStatus == "train":
    train_dataset = datasets.ImageFolder(root=root_dir)

    train_transforms = transforms.Compose(
        [
            transforms.Resize(336),
            transforms.RandomCrop(336, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3),
        ]
    )

    train_dataset.transform = train_transforms
    train_batch_size = 128

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4
    )
    num_classes = len(train_dataset.classes)

    model = col333_model(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(train_dataset))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")

    num_epochs = 85

    train_model(model, train_loader, criterion, optimizer, modelPath, num_epochs)

else:
    test_dataset = datasets.ImageFolder(root=root_dir)

    test_transforms = transforms.Compose(
        [
            transforms.Resize(336),
            transforms.CenterCrop(336),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset.transform = test_transforms

    test_batch_size = 128

    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4
    )

    model = col333_model()
    model = model.to(device)
    model.load_state_dict(torch.load(modelPath))
    model.eval()

    test_model(model, test_loader, "bird.csv")
