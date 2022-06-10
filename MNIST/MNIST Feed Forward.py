import PIL.ImageOps
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader as DataLoader
from torchvision.datasets.mnist import MNIST as MNIST
from torchvision.transforms import transforms as transforms

transform = transforms.Compose(
    [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
training_dataset = MNIST(root='../data', train=True, download=True, transform=transform)
validation_dataset = MNIST(root='../data', train=False, download=True, transform=transform)

training_loader = DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=100, shuffle=False)


def im_convert(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5,) + np.array(0.5, ))
    image = image.clip(0, 1)
    return image


class Classifier(nn.Module):
    def __init__(self, d_in, h1, h2, d_out):
        super().__init__()
        self.linear1 = nn.Linear(d_in, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, d_out)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


model = Classifier(784, 125, 65, 10)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 12

running_loss_history = []
running_corrects_history = []

val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs):
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0
    for inputs, labels in training_loader:
        inputs = inputs.view(inputs.shape[0], -1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predictions = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(predictions == labels.data)
    else:
        with torch.no_grad():
            for val_inputs, val_labels in validation_loader:
                val_inputs = val_inputs.view(val_inputs.shape[0], -1)
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_labels)
                _, predictions = torch.max(val_outputs, 1)
                val_running_loss += loss.item()
                val_running_corrects += torch.sum(predictions == val_labels.data)
        epoch_loss = running_loss / len(training_loader)
        epoch_acc = running_corrects.float() / len(training_loader)
        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc)
        val_epoch_loss = val_running_loss / len(validation_loader)
        val_epoch_acc = val_running_corrects.float() / len(validation_loader)
        val_running_loss_history.append(val_epoch_loss)
        val_running_corrects_history.append(val_epoch_acc)
        print('training loss: {:.4f}, accuracy: {:.4f}'.format(epoch_loss, epoch_acc))
        print('validation loss: {:.4f}, accuracy: {:.4f}'.format(val_epoch_loss, val_epoch_acc))

plt.plot(running_loss_history, label='training loss')
plt.show()

plt.plot(running_corrects_history, label='training corrects')
plt.show()

plt.plot(val_running_loss_history, label='validation loss')
plt.show()

plt.plot(val_running_corrects_history, label='validation corrects')
plt.show()

url = 'https://images.homedepot-static.com/productImages/007164ea-d47e-4f66-8d8c-fd9f621984a2/svn/architectural' \
      '-mailboxes-house-letters-numbers-3585b-5-64_1000.jpg '
response = requests.get(url, stream=True)
img = Image.open(response.raw)
img = PIL.ImageOps.invert(img)
img = img.convert('1')
img = transform(img)

img = img.view(img.shape[0], -1)
output = model(img)
_, prediction = torch.max(output, 1)
print(prediction.item())
