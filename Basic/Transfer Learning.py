import numpy as np
import requests
import torch
from PIL import Image
from torch import nn
from torchvision import transforms, datasets, models


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


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    classes = ('ant', 'bee')

    transform_train = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(10),
         transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
         transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )

    training_dataset = datasets.ImageFolder('../data/ants_and_bees/train', transform=transform_train)
    validation_dataset = datasets.ImageFolder('../data/ants_and_bees/val', transform=transform_train)

    training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=20, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=20, shuffle=False)

    model = models.vgg16(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad_(False)

    n_inputs = model.classifier[6].in_features
    last_layer = nn.Linear(n_inputs, len(classes))
    model.classifier[6] = last_layer
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 12

    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        validation_loss = 0.0
        for inputs, labels in training_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            training_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                validation_loss += criterion(outputs, labels).item()

        training_loss /= len(training_loader.dataset)
        validation_loss /= len(validation_loader.dataset)

        print('\nEpoch {}'.format(epoch))
        print('------------------------------------------------------------')
        print('Training Loss: {}'.format(training_loss))
        print('Validation Epoch Loss: {}'.format(training_loss))

    img = im_get('https://www.sierraclub.org/sites/www.sierraclub.org/files/styles/flexslider_full/public/sierra/articles/big/SIERRA%20Bees%20WB.jpg?itok=qTK1GSdr')

    # Convert the image to 3 color depth
    img = img.convert('RGB')

    # Transform the input image to the right size
    img = transform(img)

    # Move image to device
    image = img.to(device).unsqueeze(0)

    # Getting the model prediction
    output = model(image)
    _, prediction = torch.max(output, 1)
    print(classes[prediction.item()])


if __name__ == '__main__':
    main()
