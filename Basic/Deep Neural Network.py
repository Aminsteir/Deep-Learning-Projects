import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets


class Model(nn.Module):
    def __init__(self, input_size, h1, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, h1)
        self.linear2 = nn.Linear(h1, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        x = torch.sigmoid(self.linear2(x))
        return x

    def predict(self, x):
        return 1 if self.forward(x) >= 0.5 else 0


def scatter_plot(x, y):
    plt.scatter(x[y == 0, 0], x[y == 0, 1])
    plt.scatter(x[y == 1, 0], x[y == 1, 1])


def plot_decision_boundary(x, model):
    x_span = np.linspace(min(x[:, 0]) - 0.25, max(x[:, 0]) + 0.25)
    y_span = np.linspace(min(x[:, 1]) - 0.25, max(x[:, 1]) + 0.25)
    xx, yy = np.meshgrid(x_span, y_span)
    grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    pred_func = model.forward(grid)
    z = pred_func.view(xx.shape).detach().numpy()
    plt.contourf(xx, yy, z, alpha=0.5)


def main():
    torch.manual_seed(2)

    n_pts = 500
    x, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)
    x_data = torch.Tensor(x)
    y_data = torch.Tensor(y.reshape(n_pts, 1))

    model = Model(2, 4, 1)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    epochs = 150
    losses = []
    for i in range(1, epochs + 1):
        y_pred = model.forward(x_data)
        loss = criterion(y_pred, y_data)
        if i % 50 == 0:
            print("Epoch", i, ": Loss is", loss.item())
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(range(epochs), losses)
    plt.xlabel('Loss')
    plt.ylabel('Epoch')
    plt.show()

    scatter_plot(x, y)
    plot_decision_boundary(x, model)
    plt.show()

    p1 = torch.Tensor([0, 0])
    print("Point 1 in class {}".format(model.predict(p1)))


if __name__ == '__main__':
    main()
