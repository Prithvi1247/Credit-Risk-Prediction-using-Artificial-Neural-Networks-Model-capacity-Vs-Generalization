import torch # type:ignore
import torch.nn as nn # type:ignore

class CreditRiskANN(nn.Module):

    def __init__(self, input_size, hidden_layers):
        super().__init__()

        layers = []
        prev = input_size

        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        layers.append(nn.Linear(prev, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

