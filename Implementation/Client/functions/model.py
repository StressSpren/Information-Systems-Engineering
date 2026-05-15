import torch.nn as nn
import torch.nn.functional as F

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