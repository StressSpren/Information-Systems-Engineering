# Imports
import torch
from PIL import Image
from torchvision import transforms
from functions.ec2 import download_ec2 as dec2

# Custom Imports
from functions.model import Net


dec2('models/model.pth')
# Declaring Model
model = Net()
model.load_state_dict(torch.load('models/model.pth'))
model.eval()

mnist_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),  # invert only DIDA
    transforms.Normalize((0.1307,), (0.3081,))
])

# Prediction Function
def predict_image(path, show=True):
    model.eval()
    img = Image.open(path).convert('L')
    tensor = mnist_transform(img)
    batch = tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(batch)
        pred = output.argmax(dim=1).item()

    if show:
        disp = tensor.clone()
        disp = disp * 0.3081 + 0.1307
        print(pred)

    return pred