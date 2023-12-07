import torch
import torch.nn as nn
import torch.nn.functional as F

class KMeansPT(nn.Module):
    def __init__(self, input_size: int, K: int = 3, mask: str = 'max'):
        """
        K-means PyTorch module.

        Args:
            input_size (int): Size of input data.
            K (int, optional): Number of clusters/centroids. Defaults to 3.
            mask (str, optional): Type of masking to use for the output of the model ("max" or "softmax"). Defaults to 'max'.
        """
        super().__init__()

        # Model parameters
        self.K = K
        self.V = nn.Parameter(torch.randn([K, input_size], requires_grad=True)).float()
        self.mask = mask

    def init_centroids(self, X: torch.Tensor):
        """
        Initialize centroids using a random subset of the input data.

        Args:
            X (torch.Tensor): Input data.
        """
        # Select random elements from the dataset as initial centroids
        X2 = X.clone()
        # Perform a random permutation and select the K first
        perm = torch.randperm(X2.size(0))
        idx = perm[:self.K]
        self.V = nn.Parameter(X2[idx], requires_grad=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the K-means module.

        Args:
            X (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Matrix containing distances to centroids multiplied by the mask.
        """
        # Matrix to store distances
        l = len(X)
        D = torch.zeros((l, self.K))

        # Matrix serving as a mask
        U = torch.zeros_like(D)

        # Calculate distances to centroids
        for k in range(self.K):
            # Calculate distance to each centroid
            d = torch.squeeze(torch.cdist(X, self.V[k:k+1], 2))
            D[:, k] = d

        if self.mask == "max":
            # Only update gradients assigned to the nearest centroid
            U[torch.arange(len(U)), D.argmin(1)] = 1

        elif self.mask == "softmax":
            # Use softmax to create a mask, considering negative distance
            U = F.softmax(-D, dim=1)

        # Apply the mask
        D = D * U

        return D
