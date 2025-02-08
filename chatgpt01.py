import torch
import torch.nn as nn


class DirBatchNorm2D(nn.Module):
    def __init__(self, num_per_dir):
        super().__init__()
        self.num_per_dir = num_per_dir

        # Define 4 separate batch normalization layers explicitly
        self.bn1 = nn.BatchNorm2d(num_per_dir)
        self.bn2 = nn.BatchNorm2d(num_per_dir)
        self.bn3 = nn.BatchNorm2d(num_per_dir)
        self.bn4 = nn.BatchNorm2d(num_per_dir)

    def forward(self, x):
        N, C, H, W = x.shape
        assert C == 4 * self.num_per_dir, "Channel count must be 4 * num_per_dir"

        # Split input tensor into 4 groups along the channel dimension
        x1, x2, x3, x4 = torch.chunk(x, chunks=4, dim=1)

        # Apply batch normalization separately
        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x3 = self.bn3(x3)
        x4 = self.bn4(x4)

        # Concatenate the outputs back along the channel dimension
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x
