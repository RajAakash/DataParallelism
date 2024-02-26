import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale images
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

# Download and load the dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# First part of the model
class ModelPart1(nn.Module):
    def __init__(self):
        super(ModelPart1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        return x
    
# Second part of the model
class ModelPart2(nn.Module):
    def __init__(self):
        super(ModelPart2, self).__init__()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model parts and move them to the appropriate device
model_part1 = ModelPart1().to(device)
model_part2 = ModelPart2().to(device)

# If you have multiple GPUs, wrap model parts with DataParallel (for data parallelism)
if torch.cuda.device_count() > 1:
    model_part1 = nn.DataParallel(model_part1)
    model_part2 = nn.DataParallel(model_part2)

optimizer = optim.Adam(list(model_part1.parameters()) + list(model_part2.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model parts and move them to the appropriate device
model_part1 = ModelPart1().to(device)
model_part2 = ModelPart2().to(device)

# If you have multiple GPUs, wrap model parts with DataParallel (for data parallelism)
if torch.cuda.device_count() > 1:
    model_part1 = nn.DataParallel(model_part1)
    model_part2 = nn.DataParallel(model_part2)

optimizer = optim.Adam(list(model_part1.parameters()) + list(model_part2.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass through the first part
        part1_output = model_part1(inputs)

        # Forward pass through the second part
        outputs = model_part2(part1_output)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

print('Finished Training')