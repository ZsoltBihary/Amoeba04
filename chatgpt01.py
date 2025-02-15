import torch

# Create the original tensor (2, 4, 3, 1, 1)
x1 = torch.arange(2 * 4 * 3 * 2 * 2).reshape(2, 4, 3, 2, 2)

print("Original Tensor (x1):")
print(x1.squeeze(-1).squeeze(-1))  # Remove last 2 dims for better visualization

# Correct reshaping, NO PERMUTE NEEDED!
x2 = x1.reshape(2, 12, 2, 2)

print("\nReshaped Tensor (x2):")
print(x2.squeeze(-1).squeeze(-1))  # Again, remove last 2 dims for readability

# Verify stacking
print("\nVerifying Stacking for the first batch:")
for i in range(4):
    print(f"Block {i+1}: {x2[0, i*3:(i+1)*3, 0, 0].tolist()}")
