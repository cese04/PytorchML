import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
import torch.nn.functional as F

import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd

# Import the pytorch Kmeans class
from Models.KMeans.kmeans_torch import KMeansPT


iris = load_iris()
X = iris.data
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

# Check if GPU is available and move data/model accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Declare model and use  GPU
kmeans_model = KMeansPT(input_size=4,
                        K=3,
                        mask="softmax").to(device)

# Standardize and move vectors to GPU if available
# Since the dataset is small the whole set can be stored in the GPU memory
X_T = torch.from_numpy(X).float().to(device)

X_T = X_T - X_T.mean(0)
X_T = X_T / X_T.std(0)

# Initialize centroids
kmeans_model.init_centroids(X_T)

# Use ADAM as the optimizer
optimizer = torch.optim.Adam(kmeans_model.parameters(), lr=0.5)

# Training cycle
for i in range(10):

    # Calculate the output
    out = kmeans_model(X_T)

    # Cost function to optimize: average distance to the centroids
    # If max is the mask only those assigned to the centroid will have gradient and 0 otherwise
    l = torch.mean((out).pow(2))

    # Update the optimizer, calculate the gradients for the centroids
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

print(l.grad_fn)

# Return from GPU tensor to numpy array and re-scale the centroids
centroids = kmeans_model.V.detach().cpu().numpy() * X.std(0) + X.mean(0)
print("KMeans Centroids:\n", np.round(centroids, 2))

# Calculate real centroids for each true class
real_centroids = np.array([X[y == label].mean(axis=0) for label in range(3)])
print("Real Centroids:\n", np.round(real_centroids, 2))

# Predict cluster labels
with torch.no_grad():
    y_hat = out.argmax(1).detach().cpu().numpy()


# Plot the results
sns.set_context("talk", font_scale=1.1)
plt.figure(figsize=(12, 6))

plt.subplot(121)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_hat, palette="Dark2")
plt.title("Clusters found")

plt.subplot(122)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Dark2")

plt.title("Real classes")
plt.show()
