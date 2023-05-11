import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, n_featuremap=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_featuremap),
            nn.ReLU(),
            nn.Linear(n_featuremap, n_featuremap * 2),
            nn.ReLU(),
            nn.Linear(n_featuremap * 2, n_featuremap * 4),
            nn.ReLU(),
            nn.Linear(n_featuremap * 4, n_featuremap * 8),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_featuremap * 8, n_featuremap * 4),
            nn.ReLU(),
            nn.Linear(n_featuremap * 4, n_featuremap * 2),
            nn.ReLU(),
            nn.Linear(n_featuremap * 2, n_featuremap),
            nn.ReLU(),
            nn.Linear(n_featuremap, input_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x