import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, Dx, K, H=64):
        super().__init__()
        self.gru = nn.GRU(Dx, H, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * H, K)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.fc(h)
