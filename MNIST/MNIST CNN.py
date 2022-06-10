import PIL
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
    image = image * np.array((0.5,) + np.array(0.5, ))

    # Clips the image
    image = image.clip(0, 1)
    return image


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = func.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    def predict(self, image):
        return self.forward(image)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    training_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)

    training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)

    model = LeNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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

    img = im_get(
        'https://images.homedepot-static.com/productImages/007164ea-d47e-4f66-8d8c-fd9f621984a2/svn/architectural'
        '-mailboxes-house-letters-numbers-3585b-5-64_1000.jpg')

    # Invert Image Color
    img = PIL.ImageOps.invert(img)
    img = img.convert('1')

    # Transform the input image to the right size
    img = transform(img)

    # Move image to device
    img = img.to(device)[0].unsqueeze(0).unsqueeze(0)

    # Getting the model prediction
    output = model.predict(img)
    _, prediction = torch.max(output, 1)
    print(prediction.item())


if __name__ == '__main__':
    main()
