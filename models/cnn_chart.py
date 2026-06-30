import torch
import torch.nn as nn
from .base import BaseDeepModel


class CNNChartPatternRecognizer(BaseDeepModel):
    def __init__(self, in_channels=5, num_classes=5):
        super().__init__(name='CNN_Chart')
        self.in_channels = in_channels
        self.input_size = in_channels
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


