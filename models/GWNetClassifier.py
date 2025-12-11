from models.gwnet_modules import gwnet
import torch.nn as nn
import torch.nn.functional as F


class GWNetClassifier(nn.Module):
    def __init__(self, device, num_nodes, num_classes, **kwargs):
        super().__init__()

        self.gwnet = gwnet(
            device=device,
            num_nodes=num_nodes,
            in_dim=1,               # Average Speed is stil the only input feature
            out_dim=1,              # Since classification, only one horizon is predicted
            **kwargs
        )
        
        skip_channels = kwargs.get('skip_channels', 256)    # skip = output from TCN blocks
        end_channels = kwargs.get('end_channels', 512)      # end = extra layer between classification 
        
        # Redefine final layer as fully connected convolution, with each sensor having n classes number of final nodes
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1))
        self.end_conv_2_cls = nn.Conv2d(in_channels=end_channels, out_channels=num_classes, kernel_size=(1,1))
        
    def forward(self, input):
        # Reshape input from (B, T, N) -> (B, 1, N, T)  - only 1 input channel is needed 
        x = input.permute(0, 2, 1).unsqueeze(1) 
        
        # First Pass through GWNet normally, skip layer before sequence forecasting is returned 
        skip = self.gwnet(x)
        
        # Classification Block
        h = F.relu(skip)
        h = F.relu(self.end_conv_1(h))
        out = self.end_conv_2_cls(h)

        # out shape: (B, num_classes, N, T)
        out = self.end_conv_2_cls(h)       
        out = out.mean(dim=-1)       # remove time dimension safely

        # torch complains if contiguous left out, sometimes throws error for large windows and horizons (48w -> 48h)
        # if very large window, timestep increases in dimension
        return out.permute(0, 2, 1).contiguous()
