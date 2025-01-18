import torch
import torch.nn as nn
# import torch.nn.functional as F


class Amoeba:
    def __init__(self):
        self.encoder = self.Encoder()

    # class Encoder(nn.Module):
    #     def forward(self, state):
    #         # Example encoding logic
    #         return state.sum(dim=-1, keepdim=True)

    class Encoder(nn.Module):
        def __init__(self, args: dict):
            super(Amoeba.Encoder, self).__init__()
            # TODO: This should go to the class Amoeba ...

            self.board_size = args.get('board_size')
            self.win_length = args.get('win_length')
            self.device = args.get('CUDA_device')

            self.stones = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32, device=self.device)
            kernel_tensor = torch.ones(3, device=self.device).diag_embed().unsqueeze(2).repeat(1, 1, self.win_length)
            self.dir_conv = DirectionalConvolution(3,
                                                   3,
                                                   self.win_length,
                                                   kernel_tensor)
            self.char_list = torch.arange(-self.win_length, self.win_length + 1,
                                          device=self.device, dtype=torch.float32)

        def forward(self, state):
            board = state.view(state.shape[0], self.board_size, -1)
            point_interpreted = soft_characteristic(board, self.stones).permute(0, 3, 1, 2)
            # point_interpreted: Tensor(N, 3, board_size, board_size)

            dir_sum = self.dir_conv(point_interpreted.repeat(1, 4, 1, 1))
            dir_sum = dir_sum.permute(0, 2, 3, 1)
            dir_sum = dir_sum.reshape(*dir_sum.shape[:-1], 4, 3)
            diff = dir_sum[..., 1] - dir_sum[..., 2]
            pen1 = 100.0 * (5 - torch.sum(dir_sum, dim=-1))
            pen2 = 100.0 * (dir_sum[..., 1] * dir_sum[..., 2])
            x = diff + pen1 + pen2
            dir_interpreted = soft_characteristic(x, self.char_list)
            dir_interpreted = dir_interpreted.permute(0, 3, 4, 1, 2)
            # dir_interpreted: Tensor(N, 4, 11, board_size, board_size)

            return point_interpreted, dir_interpreted

    def test_terminal_encoded(self, encoded):
        """
        Test for terminal states using encoded representation.
        Args:
            encoded (torch.Tensor): The encoded state tensor.
        Returns:
            terminal_signal (torch.Tensor): A binary tensor indicating terminal states.
        """
        return (encoded.mean(dim=-1) > 1.0).float()  # Example logic


class AmoebaModel(nn.Module):
    def __init__(self, amoeba: Amoeba, core_model: nn.Module):
        """
        Initialize the AmoebaModel.
        Args:
            amoeba (Amoeba): The game logic instance with its encoder.
            core_model (nn.Module): The trainable core model.
        """
        super(AmoebaModel, self).__init__()
        # self.encoder = amoeba.encoder

        self.core_model = core_model
        self.amoeba = amoeba

    def forward(self, state):
        """
        For training: Compute result only.
        Args:
            state (torch.Tensor): The input state tensor with shape (N, H, W).
        Returns:
            result (torch.Tensor): The output tensor from the core model.
        """
        encoded = self.amoeba.encoder(state)
        result = self.core_model(encoded)
        return result

    def inference(self, state):
        """
        For inference: Compute result and terminal signal.
        Args:
            state (torch.Tensor): The input state tensor with shape (N, H, W).
        Returns:
            result (torch.Tensor): The output tensor from the core model.
            terminal_signal (torch.Tensor): The terminal signal tensor.
        """
        encoded = self.amoeba.encoder(state)  # Encode once
        result = self.core_model(encoded)  # Use encoded for result
        terminal_signal = self.amoeba.test_terminal_encoded(encoded)  # Directly use Amoeba's test logic
        return result, terminal_signal

    def test_terminal(self, state):
        """
        Test for terminal states directly from the input state.
        Args:
            state (torch.Tensor): The input state tensor with shape (N, H, W).
        Returns:
            terminal_signal (torch.Tensor): A binary tensor indicating terminal states.
        """
        encoded = self.amoeba.encoder(state)  # Encode the state
        terminal_signal = self.amoeba.test_terminal_encoded(encoded)  # Use Amoeba's test logic
        return terminal_signal
