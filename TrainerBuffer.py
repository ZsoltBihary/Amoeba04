import torch
from torch.utils.data import Dataset


class TrainerBuffer(Dataset):
    def __init__(self, capacity, action_size, device='cpu'):
        """
        Circular buffer to store data for training.

        Args:
            capacity (int): Maximum number of data points to store.
            action_size (int): Size of the `state` and 'policy' tensors for one data point.
            device (str or torch.device): Device to store tensors (CPU or GPU).
        """
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.index = 0  # Points to the next position to overwrite
        self.data_count = 0  # Counter to keep track of the number of added data points

        # Initialize empty tensors for each data variable
        self.state = torch.zeros((capacity, action_size), device=device)
        self.policy = torch.zeros((capacity, action_size), device=device)
        self.state_value = torch.zeros(capacity, device=device)  # state_value is scalar per data point

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx >= self.size:
            raise IndexError("Index out of range")
        return self.state[idx], self.policy[idx], self.state_value[idx]

    def reset_counter(self):
        self.data_count = 0

    def add_batch(self, states, policies, state_values):
        """
        Add a batch of new data points to the buffer.

        Args:
            states (torch.Tensor): Batch of states, shape (batch_size, position_size).
            policies (torch.Tensor): Batch of policies, shape (batch_size, position_size).
            state_values (torch.Tensor): Batch of state values, shape (batch_size, 1).
        """
        batch_size = states.shape[0]
        if policies.shape[0] != batch_size or state_values.shape[0] != batch_size:
            raise ValueError("Batch sizes of states, policies, and state_values must match.")

        # Move data to the correct CUDA_device if necessary
        states = states.to(self.device)
        policies = policies.to(self.device)
        state_values = state_values.to(self.device)

        # Calculate the split index (where the batch wraps around the circular buffer)
        end_idx = self.index + batch_size
        wrap_idx = end_idx % self.capacity

        if end_idx <= self.capacity:  # No wrapping case
            self.state[self.index:end_idx, :] = states
            self.policy[self.index:end_idx, :] = policies
            self.state_value[self.index:end_idx] = state_values
        else:  # Wrapping case
            split = self.capacity - self.index
            self.state[self.index:, :] = states[:split, :]
            self.policy[self.index:, :] = policies[:split, :]
            self.state_value[self.index:] = state_values[:split]

            self.state[:wrap_idx, :] = states[split:, :]
            self.policy[:wrap_idx, :] = policies[split:, :]
            self.state_value[:wrap_idx] = state_values[split:]

        # Update size, index and data_count
        self.index = wrap_idx
        self.size = min(self.size + batch_size, self.capacity)
        self.data_count += batch_size

        # print('Added ', batch_size, 'data points to trainer buffer')


if __name__ == "__main__":
    # Parameters
    buffer_capacity = 10  # Small capacity for demonstration
    action_size = 225  # Shape of one state

    # Initialize buffer
    buffer = TrainerBuffer(buffer_capacity, action_size, device='cpu')

    # Generate batch data
    batch_size = 6
    states = torch.randn(batch_size, action_size)
    policies = torch.randn(batch_size, action_size)
    state_values = torch.randn(batch_size)

    # Add first batch (no wrapping)
    buffer.add_batch(states, policies, state_values)
    print(f"Buffer size after first batch: {len(buffer)}")  # Output: 6
    print(f"Data count after first batch: {buffer.data_count}")  # Output: 6

    # Add another batch (causing wrapping)
    states = torch.randn(batch_size, action_size)
    policies = torch.randn(batch_size, action_size)
    state_values = torch.randn(batch_size)
    buffer.add_batch(states, policies, state_values)
    print(f"Buffer size after second batch: {len(buffer)}")  # Output: 10 (max capacity)
    print(f"Data count after second batch: {buffer.data_count}")  # Output: 12

    # Inspect buffer data
    print("State at index 0:", buffer[0][0].shape)
