import torch
import torch.nn as nn
import numpy as np
from .base import BaseDeepModel


class CNNChartPatternRecognizer(BaseDeepModel):
    def __init__(self, in_channels=5, num_classes=5):
        super().__init__(name='CNN_Chart')
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        self.pattern_classifier = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x, return_pattern=False):
        x = x.permute(0, 2, 1) if x.dim() == 3 else x
        features = self.conv_layers(x)
        features = features.squeeze(-1)
        score = self.fc(features).squeeze(-1)
        if return_pattern:
            pattern_probs = self.pattern_classifier(features)
            return score, pattern_probs
        return score

    def extract_chart_features(self, stock_df, seq_len=60):
        prices = stock_df['close'].values
        volumes = stock_df['volume'].values if 'volume' in stock_df.columns else np.ones_like(prices)

        if 'high' in stock_df.columns and 'low' in stock_df.columns:
            highs = stock_df['high'].values
            lows = stock_df['low'].values
        else:
            highs = prices * 1.02
            lows = prices * 0.98

        returns = np.diff(prices, prepend=prices[0]) / prices[0]

        features = np.column_stack([
            (prices - prices.mean()) / (prices.std() + 1e-8),
            (volumes - volumes.mean()) / (volumes.std() + 1e-8),
            returns,
            (highs - prices) / (prices + 1e-8),
            (prices - lows) / (prices + 1e-8),
        ])

        if len(features) < seq_len:
            pad = np.zeros((seq_len - len(features), features.shape[1]))
            features = np.vstack([pad, features])

        features = features[-seq_len:]
        return torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
