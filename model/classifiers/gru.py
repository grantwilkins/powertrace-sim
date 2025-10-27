import math

import torch
import torch.nn as nn


class LockedDropout(nn.Module):
    """Variational/locked dropout: same mask across time."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        # x: (B, T, D)
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        return x * mask


class GRUClassifier(nn.Module):
    def __init__(
        self,
        Dx,
        K,
        H=64,
        bidirectional=True,
        num_layers=2,
        rnn_dropout=0.2,
        head_dropout=0.2,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(Dx)
        self.lockdrop = LockedDropout(rnn_dropout)

        self.gru = nn.GRU(
            input_size=Dx,
            hidden_size=H,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0,  # we do locked dropout ourselves
        )

        out_dim = H * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(out_dim, K),
        )

        self._init_weights()

    def _init_weights(self):
        # Orthogonal for recurrent, xavier for linear
        for name, p in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p, gain=math.sqrt(2))
            elif "bias" in name:
                nn.init.zeros_(p)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T, Dx)
        x = self.in_norm(x)
        x = self.lockdrop(x)
        h, _ = self.gru(x)  # (B, T, H[*2])
        return self.head(h)  # (B, T, K)
