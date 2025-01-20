import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from ClassTrainerBuffer import TrainerBuffer


class Trainer:
    def __init__(self, model, buffer: TrainerBuffer):
        self.model = model
        self.buffer = buffer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0001,
        )
        self.policy_loss_fn = nn.CrossEntropyLoss(reduction="mean")  # Using built-in CrossEntropyLoss
        self.value_loss_fn = nn.MSELoss(reduction="mean")  # Using built-in MSELoss

    def custom_loss(self, logit, state_value, target_policy, target_state_value):
        """
        Compute the custom loss as a combination of policy loss and value loss.

        Args:
            logit: Predicted logits from the model, shape (batch_size, position_size)
            state_value: Predicted state value from the model, shape (batch_size,)
            target_policy: Target policy distribution, shape (batch_size, position_size)
            target_state_value: Target state value, shape (batch_size,)

        Returns:
            Combined loss value (policy loss + value loss).
        """
        # Policy loss using CrossEntropyLoss
        policy_loss = self.policy_loss_fn(logit, target_policy)

        # Value loss using Mean Squared Error Loss
        value_loss = self.value_loss_fn(state_value, target_state_value)

        return policy_loss + value_loss

    def improve_model(self):
        """
        Improve the model using data from the buffer by training for a few epochs.
        """
        train_loader = DataLoader(self.buffer, batch_size=64, shuffle=True)
        self.fit(train_loader, num_epochs=10)
        self.buffer.reset_counter()

    def fit(self, train_loader, num_epochs):
        """
        Train the model for a specified number of epochs.

        Args:
            train_loader: DataLoader object for the training data.
            num_epochs: Number of training epochs.
        """
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx, (state, target_policy, target_state_value) in enumerate(train_loader):
                # Move data to the same CUDA_device as the model
                state = state.to(self.model.device)
                target_policy = target_policy.to(self.model.device)
                target_state_value = target_state_value.to(self.model.device)

                # Forward pass
                logit, state_value = self.model(state)

                # Compute the loss
                loss = self.custom_loss(logit, state_value, target_policy, target_state_value)

                # Backward pass and optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate loss for reporting
                epoch_loss += loss.item()

            # Print epoch loss for monitoring
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")

# ********************* MY OLD IMPLEMENTATION **********************************
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from ClassTrainerBuffer import TrainerBuffer
# from line_profiler_pycharm import profile
#
#
# class Trainer:
#     def __init__(self, model, buffer: TrainerBuffer):
#         self.model = model
#         self.buffer = buffer
#         self.optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
#         self.policy_loss_fn = nn.CrossEntropyLoss
#         self.value_loss_fn = nn.MSELoss
#
#     def custom_loss(self, logit, state_value, target_policy, target_state_value):
#         # Please implement policy loss calculation
#         policy_loss = 0
#         # Please implement value loss calculation
#         value_loss = 0
#         return policy_loss + value_loss
#
#     def improve_model(self):
#         train_loader = DataLoader(self.buffer, batch_size=64, shuffle=True)
#         # Please implement training loop here
#         self.fit(train_loader, num_epochs=10)
#         self.buffer.reset_counter()
#
#
#
#
#     def fit(self, train_loader, num_epochs):
#         for epoch in range(num_epochs):
#             self.model.train()
#             epoch_loss = 0.0
#
#             for batch_encoded_state, batch_policy, batch_value, batch_weight in train_loader:
#                 # Zero the parameter gradients
#                 self.optimizer.zero_grad()
#                 # Forward pass
#                 policy_predicted, value_predicted = self.model(batch_encoded_state)
#                 # Compute custom loss with sample weights
#                 loss = self.custom_loss(policy_predicted, batch_policy, value_predicted, batch_value, batch_weight)
#                 # Backward pass and optimization
#                 loss.backward()
#                 self.optimizer.step()
#                 epoch_loss += loss.item()
#             # Print epoch loss
#             print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader.dataset)}')
#         print("Training complete")
#
#     # def improve_model(self):
#     #     train_loader = DataLoader(self.buffer, batch_size=64, shuffle=True)
#     #     # Please implement training loop here
#     #     self.fit(train_loader, num_epochs=10)
#     #     self.buffer.reset_counter()
#     #
#     # @profile
#     # def fit(self, train_loader, num_epochs):
#     #     for epoch in range(num_epochs):
#     #         self.model.train()
#     #         epoch_loss = 0.0
#     #
#     #         for batch_encoded_state, batch_policy, batch_value, batch_weight in train_loader:
#     #             # Zero the parameter gradients
#     #             self.optimizer.zero_grad()
#     #             # Forward pass
#     #             policy_predicted, value_predicted = self.model(batch_encoded_state)
#     #             # Compute custom loss with sample weights
#     #             loss = self.custom_loss(policy_predicted, batch_policy, value_predicted, batch_value, batch_weight)
#     #             # Backward pass and optimization
#     #             loss.backward()
#     #             self.optimizer.step()
#     #             epoch_loss += loss.item()
#     #         # Print epoch loss
#     #         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader.dataset)}')
#     #     print("Training complete")
#     #
