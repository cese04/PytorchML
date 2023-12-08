import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F

import seaborn as sns
from sklearn.datasets import make_moons, make_circles, make_s_curve, load_iris
import pandas as pd

# Import the pytorch Kmeans class
from Models.KMeans.kmeans_torch import KMeansPT


sns.set_context("talk", font_scale=1.2)

iris = load_iris()
X = iris.data[:, :4]  # we only take the first four features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
# Plot the training points
sns.scatterplot(x=X[:, 0],
                y=X[:, 1],
                hue=y,
                palette="Dark2",
                alpha=0.7)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()


sns.pairplot(pd.DataFrame({
    'sepal length': X[:, 0],
    'sepal width': X[:, 1],
    'petal length': X[:, 2],
    'petal width': X[:, 3],
    'class': y
}),
    hue='class'
)
plt.show()


# Declare model and use the GPU
kmp = KMeansPT(4, 3, mask="softmax").cuda()

# Standardize and pass the vectors to GPU
X_T = torch.from_numpy(X).float().cuda()

X_T = X_T - X_T.mean(0)
X_T = X_T / X_T.std(0)

# Initialize centroids
kmp.init_centroids(X_T)

# kmp = kmp.cuda()

# Use ADAM as the optimizer
optimizer = torch.optim.Adam(kmp.parameters(), lr=0.5)

# Training cycle
for i in range(10):

    # Calculate the output
    out = kmp(X_T)

    # Cost function to optimize, the average distance to the centroids
    l = torch.mean((out).pow(2))

    # Update the optimizer calculate the gradients for the centroids
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

print(l.grad_fn)

kmp.V.detach().cpu().numpy() * X.std(0) + X.mean(0)
print(kmp.V)

y_hat = out.argmax(1).detach().cpu().numpy()

sns.set_context("talk", font_scale=1.1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_hat, palette="Dark2")
plt.title("Clusters found")
plt.subplot(122)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Dark2")

plt.title("Real classes")
plt.show()
