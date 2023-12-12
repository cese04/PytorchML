import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import seaborn as sns

# Import the pytorch Kmeans class
from Models.KMeans.kmeans_torch import KMeansPT

N_CLUSTERS = 30
N_ITERS = 15

# Define a transform to normalize the images
transform = transforms.Compose([transforms.ToTensor(),
                                ])


# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/',
                          download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=2048*8, shuffle=True)


# Create an iterator to pass batches of data
dataiter = iter(trainloader)

# Create a batch to initiate the centroids
images_examples, labels = next(dataiter)

# Define the model and pass the images to initiate the centroids
model = KMeansPT(784,
                 N_CLUSTERS,
                 mask='max').cuda()
images_examples = images_examples.view(images_examples.shape[0], -1)
model.init_centroids(images_examples.cuda())
model.cuda()

# Use ADAM as the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)


# Training cycle
for e in range(N_ITERS):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.cuda()
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)

        # The loss for the model is the mean squared distance to the nearest cluster
        loss = torch.mean(output.pow(2))

        # Update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

clusters = model.V
clusters = clusters.detach().cpu().numpy()

szC = np.shape(clusters)

print("Cluster sizes: ", szC)

# Display the clusters
plt.figure(figsize=(8, 8))
for i in range(25):

    plt.subplot(5, 5, i+1)
    # Return from flat vector to image shape and display the clusters
    plt.imshow(
        np.clip(np.uint8(clusters[i].reshape(28, 28)*255),
                0, 255),
        cmap="gray")

plt.show()
