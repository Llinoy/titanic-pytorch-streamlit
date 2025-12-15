import torch
import torch.nn as nn


class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=[128, 64], dropout: float = 0.2):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))  # output logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # (batch,)
