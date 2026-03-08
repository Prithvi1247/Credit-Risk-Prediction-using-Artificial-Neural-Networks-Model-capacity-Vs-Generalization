import torch.nn as nn # type:ignore

class CreditRiskANN(nn.Module):

    def __init__(self, input_size, hidden_layers, dropout=0.0):
        super().__init__()

        layers = []
        prev = input_size

        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev = h

        layers.append(nn.Linear(prev, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

