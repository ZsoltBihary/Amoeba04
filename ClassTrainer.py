import torch
from torch.utils.data import DataLoader, Dataset
from line_profiler_pycharm import profile
import torch.optim as optim


class Trainer:
    def __init__(self, model, buffer):
        self.model = model
        self.buffer = buffer
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)

    @profile
    def improve_model(self):
        train_loader = DataLoader(self.buffer, batch_size=64, shuffle=True)
        self.fit(train_loader, num_epochs=10)
        self.buffer.reset_new_data_count()

    def custom_loss(self,
                    policy_predicted, target_policy,
                    value_predicted, target_value,
                    weight):

        cross_entropy = -(target_policy * (policy_predicted + 0.0000001).log()).sum(dim=1)
        weighted_cross_entropy = cross_entropy * weight
        policy_loss = weighted_cross_entropy.sum() / weight.sum()
        # Compute element-wise MSE loss
        mse_loss = (value_predicted - target_value) ** 2
        # Compute the weighted loss
        weighted_mse_loss = mse_loss * weight
        # Compute the weighted average loss
        value_loss = weighted_mse_loss.sum() / weight.sum()
        return policy_loss + value_loss

    @profile
    def fit(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_encoded_state, batch_policy, batch_value, batch_weight in train_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward pass
                policy_predicted, value_predicted = self.model(batch_encoded_state)
                # Compute custom loss with sample weights
                loss = self.custom_loss(policy_predicted, batch_policy, value_predicted, batch_value, batch_weight)
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            # Print epoch loss
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader.dataset)}')
        print("Training complete")
