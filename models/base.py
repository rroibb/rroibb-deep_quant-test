import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseDeepModel(nn.Module, ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, x):
        pass

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).cpu().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
        return self
