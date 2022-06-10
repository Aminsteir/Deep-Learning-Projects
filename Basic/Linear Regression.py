import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def get_params(model):
    [w, b] = model.parameters()
    return w[0][0].item(), b[0].item()


def plot_fit(title, model, points_x, points_y):
    w1, b1 = get_params(model)
    x1 = np.array([-30, 30])
    y1 = w1 * x1 + b1
    plt.title = title
    plt.plot(x1, y1, 'r')
    plt.plot(points_x, points_y, 'o')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()


def main():
    torch.manual_seed(1)

    x = torch.randn(100, 1) * 10
    y = x + 3 * torch.randn(100, 1)

    model = LR(1, 1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 100
    losses = []
    for i in range(1, epochs + 1):
        y_pred = model.forward(x)
        loss = criterion(y_pred, y)

        if i % 20 == 0:
            print("Epoch", i, ": Loss is", loss.item())

        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(range(epochs), losses)
    plt.xlabel('Loss')
    plt.ylabel('Epoch')
    plt.show()

    plot_fit('Trained Model', model, x, y)


if __name__ == '__main__':
    main()
