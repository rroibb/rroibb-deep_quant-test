import torch
import torch.nn as nn
from .base import BaseDeepModel


class LSTMStockPredictor(BaseDeepModel):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, bidirectional=True):
        super().__init__(name='LSTM')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.bn = nn.BatchNorm1d(hidden_size * self.num_directions)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            combined = torch.cat([h_forward, h_backward], dim=1)
        else:
            combined = h_n[-1, :, :]
        combined = self.bn(combined)
        combined = self.dropout(combined)
        out = self.fc(combined).squeeze(-1)
        return out
