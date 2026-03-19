import torch.nn as nn


class GRUClassifier(nn.Module):
    """
    GRU-based classifier for predicting power states from schedule matrices.
    This model uses a bidirectional GRU to process the input schedule matrix
    and outputs a classification over K power states.
    """

    def __init__(self, Dx, K, H=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            Dx,
            H,
            num_layers=max(1, int(num_layers)),
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * H, K)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.fc(h)
