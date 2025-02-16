import torch

batch, b = 128, 15

x = torch.randn(batch, 1, b, b)  # Your 4D tensor

# Flatten while keeping the batch dimension
x_flat = x.flatten(start_dim=1)  # Shape: (batch, b^2)

print(x_flat.shape)  # (batch, b^2)

