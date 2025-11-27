import torch.nn as nn

class TCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        self.tcn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(16, 16, 3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.tcn(x)
        x = x.squeeze(-1)
        return self.fc(x)