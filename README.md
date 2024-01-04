# LLMs for RNA Folding
The code above uses attention-based transformer architectures to predict RNA structures after folding, given their sequences as strings. For more information and background on the problem, see [my notebook](https://github.com/shahabhishek1729/RNA-Folding-Introduction) introducing the Stanford Ribonanza RNA Folding competition.

### Model
The following defines the model built and trained in the above scripts:
```py
class Model(nn.Module):
    def __init__(self, dim=320, depth=12, head_size=32, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(12, dim)
        self.pos_enc = SinusoidalPosEmb(dim=dim)
        self.transformer = nn.TransformerEncoder(
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
        pos = self.pos_enc(pos)
        x = self.emb(x)
        x = x + pos

        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.proj_out(x)

        return x
```

### Credits
This work is a modification of that originally shared by @Iafoss on [Kaggle](https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb?scriptVersionId=142566306), to use information about base-pairing probabilities (BPPs) in the model itself. This leads to a model slightly wider than the one originally shared, along with changes to preprocessing steps.
