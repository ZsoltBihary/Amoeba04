import torch
import torch.nn as nn
from line_profiler_pycharm import profile


class DirectionalAttentionLayer(nn.Module):
    def __init__(self, c, d, q):
        super().__init__()
        self.c = c
        self.d = d
        self.q = q
        e = c + d
        # Define learnable transformation matrix
        self.QKV = nn.Parameter(torch.randn(q, e, 3, 2))
        # QKV shape: (q, e, k, j), where
        # q: query-key space dimension
        # e: c+d combined block dimension
        # k: query/key/value identifying index
        # j: parallel/diagonal identifying index
        self.cuda()

    @profile
    def forward(self, x):
        batch, n, row, column = x.shape
        assert n == self.c + 4 * self.d, "Incorrect input shape"

        first_block = x[:, :self.c, :, :]
        second_block = x[:, self.c:self.c + self.d, :, :]
        third_block = x[:, self.c + self.d:self.c + 2 * self.d, :, :]
        fourth_block = x[:, self.c + 2 * self.d:self.c + 3 * self.d, :, :]
        fifth_block = x[:, self.c + 3 * self.d:, :, :]

        hor = torch.cat([first_block, second_block], dim=1)
        ver = torch.cat([first_block, third_block], dim=1)
        dia = torch.cat([first_block, fourth_block], dim=1)
        ant = torch.cat([first_block, fifth_block], dim=1)
        # shape: (b, e, r, c)

        hor_ver = torch.stack(tensors=(hor, ver), dim=-1)
        dia_ant = torch.stack(tensors=(dia, ant), dim=-1)
        # shape: (b, e, r, c, i)

        all_dir = torch.stack(tensors=(hor_ver, dia_ant), dim=-2)
        # shape: (b, e, r, c, j, i)

        qkv_all = torch.einsum('qekj,bercji->bqrckji', self.QKV, all_dir)
        # shape: (b, q, r, c, k, j, i)

        return qkv_all

        # return q_hor, q_ver, q_dia, q_ant, k_hor, k_ver, k_dia, k_ant, v_hor, v_ver, v_dia, v_ant
        # return q_hor_ver, q_dia_ant, k_hor_ver, k_dia_ant, v_hor_ver, v_dia_ant


# Example usage
batch, c, d, q, b = 800, 32, 32, 16, 15  # Example dimensions
x = torch.randn(batch, c + 4 * d, b, b).cuda()  # Example input
test_layer = DirectionalAttentionLayer(c, d, q)
for i in range(1000):
    qkv_all = test_layer(x)

print(qkv_all.shape)

# class DirectionalAttentionLayer(nn.Module):
#     def __init__(self, c, d, q):
#         super(DirectionalAttentionLayer, self).__init__()
#         self.c = c
#         self.d = d
#         self.q = q
#         e = c + d
#         # Define learnable transformation matrices
#         self.Q_par = nn.Parameter(torch.randn(q, e))
#         self.Q_dia = nn.Parameter(torch.randn(q, e))
#         self.K_par = nn.Parameter(torch.randn(q, e))
#         self.K_dia = nn.Parameter(torch.randn(q, e))
#         self.V_par = nn.Parameter(torch.randn(q, e))
#         self.V_dia = nn.Parameter(torch.randn(q, e))
#
#         self.QKV = nn.Parameter(torch.randn(q, e, 3, 2))
#         # shape: (q, e, k, j),
#         # q: query-key space dimension
#         # e: c+d combined block dimension
#         # k: query/key/value identifying index
#         # j: parallel/diagonal identifying index
#
#     def forward(self, x):
#         batch_size, n, b, _ = x.shape
#         assert n == self.c + 4 * self.d, "Incorrect input shape"
#
#         first_block = x[:, :self.c, :, :]
#         second_block = x[:, self.c:self.c + self.d, :, :]
#         third_block = x[:, self.c + self.d:self.c + 2 * self.d, :, :]
#         fourth_block = x[:, self.c + 2 * self.d:self.c + 3 * self.d, :, :]
#         fifth_block = x[:, self.c + 3 * self.d:, :, :]
#
#         hor = torch.cat([first_block, second_block], dim=1)
#         ver = torch.cat([first_block, third_block], dim=1)
#         dia = torch.cat([first_block, fourth_block], dim=1)
#         ant = torch.cat([first_block, fifth_block], dim=1)
#
#         hor_ver = torch.stack(tensors=(hor, ver), dim=-1)
#         dia_ant = torch.stack(tensors=(dia, ant), dim=-1)
#         # shape: (b, c, n, m, i)
#
#         all_dir = torch.stack(tensors=(hor_ver, dia_ant), dim=-2)
#         # shape: (b, c, n, m, j, i)
#         Q = torch.stack(tensors=(self.Q_par, self.Q_dia), dim=-1)
#         # shape: (q, c, j)
#         q_all = torch.einsum('qcj,bcnmji->bqnmji', Q, all_dir)
#         # shape: (b, q, n, m, j, i)
#
#         q_hor_ver = torch.einsum('qc,bcnmi->bqnmi', self.Q_par, hor_ver)
#         q_dia_ant = torch.einsum('qc,bcnmi->bqnmi', self.Q_dia, dia_ant)
#
#         err0 = torch.max(torch.abs(q_hor_ver - q_all[..., 0, :]))
#         print("err0 = ", err0)
#
#         k_hor_ver = torch.einsum('qc,bcnmi->bqnmi', self.K_par, hor_ver)
#         k_dia_ant = torch.einsum('qc,bcnmi->bqnmi', self.K_dia, dia_ant)
#
#         v_hor_ver = torch.einsum('qc,bcnmi->bqnmi', self.V_par, hor_ver)
#         v_dia_ant = torch.einsum('qc,bcnmi->bqnmi', self.V_dia, dia_ant)
#
#         # q_hor = torch.einsum('qc,bcnm->bqnm', self.Q_par, hor)
#         # q_ver = torch.einsum('qc,bcnm->bqnm', self.Q_par, ver)
#         # q_dia = torch.einsum('qc,bcnm->bqnm', self.Q_dia, dia)
#         # q_ant = torch.einsum('qc,bcnm->bqnm', self.Q_dia, ant)
#         #
#         # k_hor = torch.einsum('qc,bcnm->bqnm', self.K_par, hor)
#         # k_ver = torch.einsum('qc,bcnm->bqnm', self.K_par, ver)
#         # k_dia = torch.einsum('qc,bcnm->bqnm', self.K_dia, dia)
#         # k_ant = torch.einsum('qc,bcnm->bqnm', self.K_dia, ant)
#         #
#         # v_hor = torch.einsum('qc,bcnm->bqnm', self.V_par, hor)
#         # v_ver = torch.einsum('qc,bcnm->bqnm', self.V_par, ver)
#         # v_dia = torch.einsum('qc,bcnm->bqnm', self.V_dia, dia)
#         # v_ant = torch.einsum('qc,bcnm->bqnm', self.V_dia, ant)
#
#         # return q_hor, q_ver, q_dia, q_ant, k_hor, k_ver, k_dia, k_ant, v_hor, v_ver, v_dia, v_ant
#         return q_hor_ver, q_dia_ant, k_hor_ver, k_dia_ant, v_hor_ver, v_dia_ant
#
