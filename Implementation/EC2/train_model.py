# PYTORCH IMPORTS

import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

# Optional Imports
# from torchvision.transforms import GaussianBlur

mnist_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dida_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),
    transforms.Normalize((0.1307,), (0.3081,))
])

# DATASET LOADING

dida_data = ImageFolder(root='./datasets/dida/70000', transform=dida_transform)
mnist_data = datasets.MNIST(root='./datasets/data', train=True, download=True, transform=mnist_transform)
mnist_test = datasets.MNIST(root='./datasets/data', train=False, download=True, transform=mnist_transform)

# Future enhancement: Include scanned images from OCR processing
# scanned_data = ImageFolder(root='./predicted_images/', transform=dida_transform)

# DATASET SPLITTING AND COMBINATION

generator = torch.Generator().manual_seed(42)
dida_train, dida_test = random_split(dida_data, [0.8, 0.2], generator=generator)
train_data = ConcatDataset([mnist_data, dida_train])
test_data = ConcatDataset([mnist_test, dida_test])

# DATA LOADER CONFIGURATION

loaders = {
    'train': DataLoader(train_data, batch_size=100, shuffle=True),
    'test': DataLoader(test_data, batch_size=100, shuffle=True)
}

# CONVOLUTIONAL NEURAL NETWORK

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        conv1_output = self.conv1(x)
        pooled1 = F.max_pool2d(conv1_output, 2) 
        x = F.relu(pooled1)
        conv2_output = self.conv2(x)
        dropout_output = self.dropout1(conv2_output)
        pooled2 = F.max_pool2d(dropout_output, 2)
        x = F.relu(pooled2)
        x = x.view(-1, 320)
        fc1_output = self.fc1(x)
        x = F.relu(fc1_output)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return x

# HARDWARE OPTIMIZATION AND MODEL SETUP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# TRAINING FUNCTION

def train(epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

       
        if batch_idx % 20 == 0:
            print('[Epoch: {} {}/{} ({:.0f}%)] [Loss: {:.6f}]'.format(
                epoch,
                batch_idx * len(data),
                len(loaders['train'].dataset),
                100. * batch_idx / len(loaders['train']),
                loss.item()))

# TESTING FUNCTION

def test():
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_size = data.size(0)
            loss = criterion(output, target)
            test_loss += loss.item() * batch_size
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += batch_size

    test_loss /= total

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        total,
        100. * correct / total))

# MAIN TRAINING LOOP

print("Starting training...")
for epoch in range(1, 11):
    print(f"\n--- Epoch {epoch}/10 ---")
    train(epoch)
    test()

print("Training completed successfully!")

# MODEL PERSISTENCE

model_path = "./models/model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")