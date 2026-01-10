import math
import statistics
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from models.Layers import RevIN, Attention_MLP
from layers.Embed import Emb


class FourierBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_node, topk=None, padding=1.0, hid_dim=128):
        super(FourierBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_node = num_node
        self.hid_dim = hid_dim
        self.tk = topk
        self.padding = padding
        g_in = (padding + 1) * in_channel // 2 + 1 + in_channel // 2 + 1

        self.attn = Attention_MLP(g_in, in_channel // 2 + 1, hidden=self.hid_dim, channel=self.num_node)

    def forward(self, x):
        B, L, N = x.size()
        # 确保新创建的 zero 张量在和 x 相同的设备上
        zero = torch.zeros(B, int(self.padding * L), N, device=x.device)
        x_fft = fft.rfft(x, dim=1)

        # fence
        x_pad = torch.cat([x, zero], dim=1)
        x_p_fft = fft.rfft(x_pad, dim=1)

        # top_amp
        eps = 1e-6
        amp_p = torch.sqrt((x_p_fft.real + eps).pow(2) + (x_p_fft.imag + eps).pow(2))
        amp = torch.sqrt((x_fft.real + eps).pow(2) + (x_fft.imag + eps).pow(2))  # 计算振幅

        topk = torch.topk(amp, self.tk, 1)
        idx = topk.indices
        values = topk.values

        # 确保 amp_topk 在正确的设备上
        amp_topk = torch.zeros_like(amp, device=amp.device)
        amp_topk = amp_topk.scatter(1, idx, values)

        x_gate = torch.cat([amp_topk, amp_p], dim=1)

        out_fft = self.attn(x_gate, x_fft, True)

        out = fft.irfft(out_fft, n=self.in_channel, dim=1)

        return out


class TimeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_node, padding, chunk_size, hid_dim=128, ratio=0.3):
        super(TimeBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_node = num_node
        self.chunk_size = chunk_size
        self.hid_dim = hid_dim
        assert (in_channel % chunk_size == 0)
        self.num_chunks = in_channel // chunk_size
        self.padding = padding
        self.ratio = ratio
        num_sel = math.floor(ratio * in_channel * padding)
        g_in = num_sel + chunk_size + in_channel
        self.attn = Attention_MLP(g_in, self.in_channel, hidden=self.hid_dim, channel=self.num_node)

    def forward(self, x):
        B, L, N = x.size()

        x1 = x.reshape(B, self.chunk_size, self.num_chunks, N)
        x1 = x1[:, :, 0, :]
        x1 = x1.permute(0, 2, 1)

        # 插值
        x2 = F.interpolate(x.permute(0, 2, 1), scale_factor=self.padding + 1, mode='linear')
        x2_int = x2[:, :, 1::2]

        shape = x2_int.shape
        num_sel = math.floor(self.ratio * shape[-1])

        # --- 修改点：确保随机生成的索引和 tensor 在同一个设备上 ---
        tensor = torch.zeros(shape, device=x.device)

        # 使用 device=x.device 确保随机排列在 GPU 上生成
        indices_to_set = torch.randperm(shape[-1], device=x.device)[:num_sel]
        tensor[..., indices_to_set] = 1

        # shuffle 同样要在 GPU 上进行
        shuffled_indices = torch.randperm(shape[-1], device=x.device)
        shuffled_tensor = tensor[..., shuffled_indices]

        result_matrix = x2_int.masked_select(shuffled_tensor.bool()).reshape(B, N, -1)

        x_gate = torch.cat([x1, x.permute(0, 2, 1), result_matrix], dim=-1)

        out = self.attn(x_gate.permute(0, 2, 1), x)

        return out


class MResBlock(nn.Module):
    def __init__(self, lookback, lookahead, hid_dim, num_node, dropout=0.1, chunk_size=40, c_dim=40):
        super(MResBlock, self).__init__()
        self.lookback = lookback
        self.lookahead = lookahead
        self.chunk_size = chunk_size
        self.num_chunks = lookback // chunk_size
        self.hid_dim = int(hid_dim)
        self.num_node = int(num_node)
        self.c_dim = int(c_dim)
        self.tk = 8
        self.padding_t = 1
        self.padding_f = 1
        self.dropout = dropout
        self._build()

    def _build(self):
        # 初始化子模块，model.cuda() 时会自动将它们移到 GPU
        self.fb = FourierBlock(self.lookback, self.lookahead, self.num_node, self.tk, self.padding_f, self.hid_dim)
        self.tb = TimeBlock(self.lookback, self.lookahead, self.num_node, self.padding_t, self.chunk_size, self.hid_dim)

    def forward(self, x):
        fout = self.fb(x)
        tout = self.tb(x)
        return fout, tout


class TFMRN(nn.Module):
    def __init__(self, lookback, lookahead, hid_dim, num_node, dropout=0, chunk_size=40, c_dim=40):
        super(TFMRN, self).__init__()
        self.lookback = int(lookback)
        self.lookahead = int(lookahead)
        self.chunk_size = int(chunk_size)
        self.num_chunks = lookback // chunk_size

        self.hid_dim = int(hid_dim)
        self.num_node = int(num_node)
        self.c_dim = int(c_dim)
        self.dropout = dropout

        self.pos_enc_scale = 0.3  # 这是一个 float，不需要 device

        self._build()

    def _build(self):
        # 这里的子模块定义是正确的，使用了 nn.Sequential 和 nn.Linear
        # model.cuda() 会自动处理它们
        self.revinlayer = RevIN(num_features=self.num_node)

        self.layer = MResBlock(
            lookback=self.lookback,
            lookahead=self.lookahead,
            hid_dim=self.hid_dim,
            num_node=self.num_node,
            chunk_size=self.chunk_size)

        self.out_proj = nn.Linear(self.lookback * 2, self.lookahead)

        self.mlp_time = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.lookback * 2, self.lookback),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.emb = Emb(self.lookback, self.hid_dim)

        self.mlp_channel = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.num_node, self.num_node),
            nn.Dropout(0.1)
        )

        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.hid_dim, self.hid_dim * 2),
            nn.GELU(),
            nn.Linear(self.hid_dim * 2, self.lookback * 2),
            nn.Dropout(0.1)
        )

        self.linear = nn.Linear(self.lookback, self.lookback * 2)

    def forward(self, x):
        # x 应该已经在 GPU 上了
        x = self.revinlayer(x, mode='norm')
        fout, tout = self.layer(x)
        out_1 = torch.cat([fout, tout], dim=1)

        out_time = out_1.transpose(1, 2)
        out_time = self.mlp_time(out_time)  # mlp_time 参数已在 GPU

        out_time_2 = self.linear(out_time)
        out_time_1 = self.emb(out_time)

        out_time_3 = out_time_1.transpose(2, 3)
        out_time_3 = self.mlp_channel(out_time_3)

        A, B, C, D = out_time_3.shape
        out_time_3 = out_time_3.reshape(B, D, A * C)

        out_time_3 = self.mlp(out_time_3)

        x = self.linear(x.transpose(1, 2))

        # 所有的组件（out_time_2, out_1, out_time_3, x）都在 GPU 上，计算安全
        out = (out_time_2 + out_1.transpose(1, 2) + self.pos_enc_scale * out_time_3) / (2 + self.pos_enc_scale)
        out = (out + x) / 2

        out = self.out_proj(out)
        out = self.revinlayer(out.permute(0, 2, 1), mode='denorm')
        return out