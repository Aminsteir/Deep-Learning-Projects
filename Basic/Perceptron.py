import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def predict(self, x):
        return 1 if self.forward(x) >= 0.5 else 0


def get_params(model):
    [w, b] = model.parameters()
    w1, w2 = w.view(2)
    return w1.item(), w2.item(), b[0].item()


def scatter_plot(x, y):
    plt.scatter(x[y == 0, 0], x[y == 0, 1])
    plt.scatter(x[y == 1, 0], x[y == 1, 1])


def plot_fit(title, model):
    plt.title = title
    w1, w2, b1 = get_params(model)
    x1 = np.array([-2.0, 2.0])
    x2 = (w1 * x1 + b1) / -w2
    plt.plot(x1, x2, 'r')


def main():
    torch.manual_seed(2)

    n_pts = 100
    centers = [[-0.5, 0.5], [0.5, -0.5]]
    x, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)
    x_data = torch.Tensor(x)
    y_data = torch.Tensor(y.reshape(100, 1))

    model = Model(2, 1)

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 5000
    losses = []
    for i in range(1, epochs + 1):
        y_pred = model.forward(x_data)
        loss = criterion(y_pred, y_data)
        if i % 1000 == 0:
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
    plot_fit('Trained Model', model)
    plt.show()

    p1 = torch.Tensor([1.0, -1.0])
    print("Point 1 in class {}".format(model.predict(p1)))


if __name__ == '__main__':
    main()
