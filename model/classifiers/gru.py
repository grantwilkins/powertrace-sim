import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, Dx, K, H=64, bidirectional=True):
        super().__init__()
        self.gru = nn.GRU(Dx, H, batch_first=True, bidirectional=bidirectional)
        gru_output_size = 2 * H if bidirectional else H
        self.fc = nn.Linear(gru_output_size, K)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.fc(h)
