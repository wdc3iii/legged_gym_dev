import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_dim, num_units, num_layers, activation, final_activation=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, num_units), activation])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(num_units, num_units))
            self.layers.append(activation)
        self.layers.append(nn.Linear(num_units, output_dim))
        if final_activation is not None:
            self.layers.append(final_activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


