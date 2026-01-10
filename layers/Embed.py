import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]





class EmbLayer(nn.Module):

    def __init__(self, patch_len, patch_step, seq_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step = patch_step

        patch_num = int((seq_len - patch_len) / patch_step + 1)
        self.d_model = d_model // patch_num
        self.ff = nn.Sequential(
            nn.Linear(patch_len, self.d_model),
        )
        self.flatten = nn.Flatten(start_dim=-2)

        self.ff_1 = nn.Sequential(
            nn.Linear(self.d_model * patch_num, d_model),
        )

    def forward(self, x):

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step) # 对输入张量 x 沿着最后一个维度（通常是时间序列或序列长度维度）
                                                                              # 进行“展开”操作，生成一系列固定长度的子序列（patch）128*7*3*48
        x = self.ff(x)   # 将展开后的张量 x 输入到 self.ff（一个线性变换的 nn.Sequential 模块）中，对每个子序列进行特征变换。128*7*3*21
        x = self.flatten(x)  # 将张量 x 沿着最后两个维度（patch_num 和 d_model）展平，合并为一个维度。128*7*63
        x = self.ff_1(x)  # 将展平后的张量 x 输入到 self.ff_1（另一个线性变换的 nn.Sequential 模块），进行进一步的特征变换128*7*64
        return x


class Emb(nn.Module):

    def __init__(self, seq_len, d_model, patch_len=[48, 24, 12, 6]):
        super().__init__()
        patch_step = patch_len
        d_model = d_model//4
        self.EmbLayer_1 = EmbLayer(patch_len[0], patch_step[0] // 2, seq_len, d_model)
        self.EmbLayer_2 = EmbLayer(patch_len[1], patch_step[1] // 2, seq_len, d_model)
        self.EmbLayer_3 = EmbLayer(patch_len[2], patch_step[2] // 2, seq_len, d_model)
        self.EmbLayer_4 = EmbLayer(patch_len[3], patch_step[3] // 2, seq_len, d_model)
        self.positional_embedding = PositionalEmbedding(d_model=d_model, max_len=seq_len)

    def forward(self, x):
        s_x1 = self.EmbLayer_1(x)
        s_x2 = self.EmbLayer_2(x)
        s_x3 = self.EmbLayer_3(x)
        s_x4 = self.EmbLayer_4(x)
        pos_enc = self.positional_embedding(x)
        s_x1 = s_x1 + pos_enc.expand(s_x1.size(0), -1, -1)  # 扩展 batch 维度
        s_x2 = s_x2 + pos_enc.expand(s_x2.size(0), -1, -1)
        s_x3 = s_x3 + pos_enc.expand(s_x3.size(0), -1, -1)
        s_x4 = s_x4 + pos_enc.expand(s_x4.size(0), -1, -1)
        #s_out = torch.cat([s_x1, s_x2, s_x3, s_x4], -1)
        s_out = torch.stack([s_x1 , s_x2, s_x3, s_x4], dim=0)
        return s_out




