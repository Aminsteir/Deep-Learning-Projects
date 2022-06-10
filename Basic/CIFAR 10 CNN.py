import numpy as np
import requests
import torch
import torch.nn.functional as func
from PIL import Image
from torch import nn
from torchvision import transforms, datasets


def im_get(url):
    response = requests.get(url, stream=True)
    return Image.open(response.raw)


def im_convert(tensor):
    # Converts the image to numpy array
    image = tensor.cpu().clone().detach().numpy().transpose(1, 2, 0)

    # Apply some transformation
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))

    # Clips the image
    image = image.clip(0, 1)
    return image


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)

        self.fc1 = nn.Linear(4 * 4 * 64, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        v = func.relu(self.conv1(x))
        v = func.max_pool2d(v, 2, 2)

        v = func.relu(self.conv2(v))
        v = func.max_pool2d(v, 2, 2)

        v = func.relu(self.conv3(v))
        v = func.max_pool2d(v, 2, 2)

        v = v.view(-1, 4 * 4 * 64)

        v = func.relu(self.fc1(v))
        v = self.dropout1(v)
        v = self.fc2(v)

        return v

    def predict(self, image):
        return self.forward(image)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(10),
         transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    training_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

    training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)

    model = LeNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 12

    for e in range(1, epochs + 1):
        epoch_loss = 0.0
        for inputs, labels in training_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.predict(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss /= len(training_loader)
        print('Epoch {} Loss: {}'.format(e, epoch_loss))

    img = im_get('https://www.cdc.gov/healthypets/images/pets/cute-dog-headshot.jpg')

    # Convert the image to 3 color depth
    img = img.convert('RGB')

    # Transform the input image to the right size
    img = transform(img)

    # Move image to device
    image = img.to(device).unsqueeze(0)

    # Getting the model prediction
    output = model.predict(image)
    _, prediction = torch.max(output, 1)
    print(classes[prediction.item()])


if __name__ == '__main__':
    main()
