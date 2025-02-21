import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueHead(nn.Module):
    def __init__(self, in_channels, board_size, num_heads, attention_dim):
        super(PolicyValueHead, self).__init__()
        self.b = board_size
        self.num_heads = num_heads

        # Learnable feature mixing instead of rigid slicing
        self.shared_transform = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Allocate channels for policy & attention
        self.conv_policy = nn.Conv2d(in_channels, 1, kernel_size=1)  # (batch, 1, b, b)
        self.conv_attention_logit = nn.Conv2d(in_channels, num_heads, kernel_size=1)  # (batch, heads, b, b)
        self.conv_space_value = nn.Conv2d(in_channels, num_heads, kernel_size=1)  # (batch, heads, b, b)

        # Combine heads into single state_value
        self.head_combine = nn.Linear(num_heads, 1)

        # Residual coupling from attention_logit → policy_logit
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Learnable scaling factor

    def forward(self, x):
        batch_size = x.shape[0]

        # Shared transformation for better feature organization
        x = self.shared_transform(x)

        # Compute policy logits
        policy_logit = self.conv_policy(x).view(batch_size, -1)  # (batch, b^2)

        # Compute attention logits
        attention_logit = self.conv_attention_logit(x).view(batch_size, self.num_heads, -1)  # (batch, heads, b^2)
        attention = F.softmax(attention_logit, dim=-1)  # Normalize across board positions

        # Compute space values
        space_value = self.conv_space_value(x).view(batch_size, self.num_heads, -1)  # (batch, heads, b^2)

        # Compute per-head state values
        state_values_per_head = torch.sum(attention * space_value, dim=-1)  # (batch, heads)

        # Aggregate across heads
        state_value = self.head_combine(state_values_per_head).squeeze(-1)  # (batch,)

        # Residual connection: Inject attention importance into policy logit
        policy_logit = policy_logit + self.alpha * attention_logit.mean(dim=1)  # (batch, b^2)

        return policy_logit, state_value

# Example usage:
batch_size = 32
board_size = 9  # 9x9 board
in_channels = 64  # CNN feature channels
num_heads = 4  # Attention heads
attention_dim = 16  # Dimensionality of attention space

# Create the model
policy_value_head = PolicyValueHead(in_channels, board_size, num_heads, attention_dim)

# Example input tensor (batch, features, b, b)
cnn_output = torch.rand(batch_size, in_channels, board_size, board_size)

# Compute policy logits and state value
policy_logit, state_value = policy_value_head(cnn_output)
print("Policy Logit Shape:", policy_logit.shape)  # Expected: (batch, b^2)
print("State Value Shape:", state_value.shape)  # Expected: (batch,)






#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class AttentionStateValueHead(nn.Module):
#     def __init__(self, in_channels, board_size):
#         super(AttentionStateValueHead, self).__init__()
#         self.b = board_size  # Board size (b x b)
#         self.in_channels = in_channels
#
#         # Pointwise convolutions (1x1) to get space_value and attention_logit
#         self.conv_space_value = nn.Conv2d(in_channels, 1, kernel_size=1)  # (batch, 1, b, b)
#         self.conv_attention_logit = nn.Conv2d(in_channels, 1, kernel_size=1)  # (batch, 1, b, b)
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#
#         # Compute space_value and attention_logit
#         space_value = self.conv_space_value(x).view(batch_size, -1)  # (batch, b^2)
#         attention_logit = self.conv_attention_logit(x).view(batch_size, -1)  # (batch, b^2)
#
#         # Softmax over the board to get attention scores
#         attention = F.softmax(attention_logit, dim=1)  # (batch, b^2)
#
#         # Compute state_value as weighted sum of space_value
#         state_value = torch.sum(attention * space_value, dim=1)  # (batch,)
#
#         return state_value
#
#
# # Example usage:
# batch_size = 32
# board_size = 9  # Gomoku uses 9x9 or 15x15 boards
# in_channels = 64  # Number of features from CNN
#
# # Create the model
# state_value_head = AttentionStateValueHead(in_channels, board_size)
#
# # Example input tensor (batch, features, b, b)
# cnn_output = torch.rand(batch_size, in_channels, board_size, board_size)
#
# # Compute state value
# state_value = state_value_head(cnn_output)
# print("State Value Shape:", state_value.shape)  # Expected: (batch_size,)
