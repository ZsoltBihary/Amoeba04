import torch
import torch.nn as nn


# Base classes without abstract methods
class BaseEncoder(nn.Module):
    def forward(self, state):
        # Default behavior, could be overridden in subclasses
        raise NotImplementedError("Subclasses should implement this method.")


class BaseCoreModel(nn.Module):
    def forward(self, encoded):
        # Default behavior, could be overridden in subclasses
        raise NotImplementedError("Subclasses should implement this method.")


# General Model Class
class RLModel(nn.Module):
    def __init__(self, encoder: BaseEncoder, core_model: BaseCoreModel):
        super(RLModel, self).__init__()
        self.encoder = encoder
        self.core_model = core_model

    def encode(self, state):
        return self.encoder(state)

    def forward(self, state):
        """For training: Compute result only."""
        encoded = self.encode(state)
        result = self.core_model(encoded)
        return result

    def inference(self, state):
        """For inference: Compute result and terminal signal."""
        encoded = self.encode(state)
        result = self.core_model(encoded)
        terminal_signal = self.test_terminal(encoded)
        return result, terminal_signal

    def test_terminal(self, encoded):
        """Logic for detecting terminal states."""
        terminal_signal = (encoded.mean(dim=-1) > 1.0).float()  # Example logic
        return terminal_signal


class SimpleEncoder(BaseEncoder):
    def forward(self, state):
        # Custom encoding logic
        return state * 2


class SimpleCoreModel(BaseCoreModel):
    def forward(self, encoded):
        # Custom core model logic
        return encoded + 1


# Create the model
encoder = SimpleEncoder()
core_model = SimpleCoreModel()

model = RLModel(encoder, core_model)

# Example state
state = torch.tensor([1.0, 2.0, 3.0])

# Inference
result, terminal_signal = model.inference(state)
print("Result:", result)
print("Terminal Signal:", terminal_signal)
