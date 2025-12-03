import torch.nn as nn


class VanillaEncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model=512,
        d_ff=2048,
        activation=nn.ReLU,
        dropout=0.1,
        batchnorm=False,
    ):

        super(VanillaEncoderLayer, self).__init__()

        self.attention = attention
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation(),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)
        self.batchnorm = batchnorm

        if batchnorm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):

        x_new, attn = self.attention(x, x, x, attn_mask)

        if self.batchnorm:
            x = (x + self.dropout(x_new)).transpose(1, 2)
            x = self.norm1(x).transpose(1, 2)
        else:
            x = self.norm1(x + self.dropout(x_new))

        x = x + self.dropout(self.feedforward(x))
        if self.batchnorm:
            x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.norm2(x)

        return x, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers):
        super(Encoder, self).__init__()
        self.encoders = nn.ModuleList(attn_layers)

    def forward(self, x, attn_mask=None):
        # B, L, D = x.shape
        attns = []
        for enc in self.encoders:
            x, attn = enc(x, attn_mask)
            attns.append(attn)

        return x, attns
