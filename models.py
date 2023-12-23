import torch
from torch import nn
from torch.nn import functional as F
from fastai.vision.all import *


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RNA_Model(nn.Module):
    def __init__(self, dim=320, depth=12, head_size=32, **kwargs):
        super().__init__()
        #         self.emb = nn.Embedding(4,dim)
        self.emb = nn.Embedding(12, dim)
        #         print("I'm about to create RoPE embeds")
        #         self.pos_enc = RotaryPEMuliHeadAttention(heads=dim//head_size, d_model=dim)
        #         self.pos_enc = RotaryPositionalEmbeddings(dim=dim)
        self.pos_enc = SinusoidalPosEmb(dim=dim)
        self.transformer = nn.TransformerEncoder(
            #             nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=dim // head_size,
                dim_feedforward=12 * dim,
                dropout=0.1,
                activation=nn.GELU(),
                batch_first=True,
                norm_first=True,
            ),
            depth,
        )
        self.proj_out = nn.Linear(dim, 2)

    def forward(self, x0):
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax]
        x = x0["seq"][:, :Lmax]

        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        #         print(f"By this point we have a shape of {pos.shape}")
        pos = self.pos_enc(pos)
        x = self.emb(x)
        x = x + pos

        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.proj_out(x)

        return x


def loss(pred, target):
    p = pred[target["mask"][:, : pred.shape[1]]]
    y = target["react"][target["mask"]].clip(0, 1)
    loss = F.l1_loss(p, y, reduction="none")
    loss = loss[~torch.isnan(loss)].mean()

    return loss


class MAE(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.x, self.y = [], []

    def accumulate(self, learn):
        x = learn.pred[learn.y["mask"][:, : learn.pred.shape[1]]]
        y = learn.y["react"][learn.y["mask"]].clip(0, 1)
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):
        x, y = torch.cat(self.x, 0), torch.cat(self.y, 0)
        loss = F.l1_loss(x, y, reduction="none")
        loss = loss[~torch.isnan(loss)].mean()
        return loss
