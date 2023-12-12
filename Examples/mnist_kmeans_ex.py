import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import seaborn as sns

# Import the pytorch Kmeans class
from Models.KMeans.kmeans_torch import KMeansPT


# Define a transform to normalize the images
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/',
                          download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=2048*10, shuffle=True)


# Create an iterator to pass batches of data
dataiter = iter(trainloader)

images_examples, labels = next(dataiter)
model = KMeansPT(784, 30, mask='max').cuda()
images_examples = images_examples.view(images_examples.shape[0], -1)
model.init_centroids(images_examples.cuda())

optimizer = optim.Adam(model.parameters(), lr=0.05)


epochs = 8
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector, pasar a CUDA
        images = images.cuda()
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = torch.mean(output.pow(2))

        # Update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

clusters = model.V.detach().cpu().numpy()

szC = np.shape(clusters)

print("CLuster sizes: ", szC)

# Display the clusters
plt.figure(figsize=(8, 8))
for i in range(10):
    plt.subplot(4, 4, i+1)
    plt.imshow(clusters[i].reshape(28, 28), cmap="gray")

plt.show()
