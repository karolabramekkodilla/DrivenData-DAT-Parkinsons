from torch import nn
class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.45),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x.squeeze(1)

class Simple3DCNNLarge(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(
                in_channels=8,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x.squeeze(1)

class ProfileMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 1024),
            nn.LayerNorm(1024),
            nn.Mish(),
            nn.Dropout(0.3),

            nn.Linear(1024, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Dropout(0.3),

            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.Mish(),
            nn.Dropout(0.2),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)