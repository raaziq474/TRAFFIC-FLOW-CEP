import torch
import torch.nn as nn


class MultiSensorTCN(nn.Module):
    """
    TCN adapted for classification of multi variate input.
    One TCN per node is used, and stacked together for a final 
    multivariate output. All TCNs are independant from each other.
    
    Input: (B, T, N) - batch, time steps, nodes/sensors
    Output: (B, N, num_classes) - batch, nodes, classification logits
    """

    def __init__(self, num_nodes, num_classes, **kwargs):
        super().__init__()

        self.num_sensors = num_nodes
        
        self.tcns = nn.ModuleList([
            TCN(num_classes, **kwargs)
            for _ in range(num_nodes)
        ])

    def forward(self, x):

        B, T, N = x.shape
        outputs = []
        
        for i in range(N):
            sensor_ts = x[:, :, i]                  # (B, T)
            sensor_ts = sensor_ts.unsqueeze(1)      # (B, 1, T)
            
            out_i = self.tcns[i](sensor_ts)         # (B, num_classes)
            outputs.append(out_i)

        # Stack each TCN output to produce predictions for all sensors 
        return torch.stack(outputs, dim=1)          # (B, N, num_classes)


class TCN(nn.Module):
    """Basic classification TCN"""

    def __init__(self, num_classes, hidden_size=32, kernel_size=3, dropout=0.0):
        super().__init__()

        self.tcn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
 
            nn.Conv1d(16, 32, kernel_size=kernel_size, padding=2, dilation=2),
            nn.ReLU(),

            nn.Conv1d(32, 32, kernel_size=kernel_size, padding=4, dilation=4),
            nn.ReLU(),
        )

        # Classfication using 1D convolutions
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):

        h = self.tcn(x)
        h = self.pool(h).squeeze(-1)    # (B, 32)
        h = self.dropout(h)             # Apply dropout before classifier
        return self.fc(h)               # (B, num_classes)



