import torch
import torch.nn as nn
from models.dgcrn_modules import DGCRN


class DGCRN_Classifier(DGCRN):
    """
    DGCRN adapted for classification, reuses GCN and GRU step logic
    from the parent DGCRN, but modifies final layer.
    
    Input: (B, T, N) - batch, time steps, nodes/sensors
    Output: (B, N, num_classes) - batch, nodes, classification logits
    """

    def __init__(self,
                 num_nodes,
                 num_classes,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 gcn_depth=2,
                 dropout=0.3,
                 subgraph_size=20,
                 node_dim=40,
                 middle_dim=2,
                 seq_length=None,  # T will be inferred from input, pass a placeholder like 12
                 rnn_size=64,
                 hyperGNN_dim=16,
                 predefined_A=None,
                 list_weight=[0.05, 0.95, 0.95],
                 tanhalpha=3):
        
        # DGCRN expects predefined adj or randomly initialises it
        if predefined_A is None:
            predefined_A_init = [torch.eye(num_nodes).to(device),torch.eye(num_nodes).to(device)]
        else:
            predefined_A_init = predefined_A

        # DGCRN expects a defined seq_length, defualt is 12 but only 1 is used 
        seq_length = seq_length if seq_length is not None else 12 

        super(DGCRN_Classifier, self).__init__(
            gcn_depth=gcn_depth,
            num_nodes=num_nodes,
            device=device,
            predefined_A=predefined_A_init,
            dropout=dropout,
            subgraph_size=subgraph_size,
            node_dim=node_dim,
            middle_dim=middle_dim,
            seq_length=seq_length,
            in_dim=1,                       # Single feature input (e.g., speed)
            out_dim=1,                      # Placeholder, will be replaced
            list_weight=list_weight,
            tanhalpha=tanhalpha,
            rnn_size=rnn_size,
            hyperGNN_dim=hyperGNN_dim
        )   # couldjust use **kwargs :|
        
        # Classification layer
        self.num_classes = num_classes
        self.fc_final = nn.Linear(self.hidden_size, self.num_classes)

        # Cannot use curriculum learning as there is only one horizon
        self.use_curriculum_learning = False
        

    def forward(self, x):

        B, T, N = x.shape
        self.seq_length = T
        
        # Transform input to base DGCRN's expected format: (B, in_dim, N, T)
        x_reshaped = x.unsqueeze(-1)
        x_transformed = x_reshaped.permute(0, 3, 2, 1)
        
        # Same update to base DGCRN
        Hidden_State, Cell_State = self.initHidden(B * self.num_nodes, self.hidden_size)

        for i in range(T):
            
            x_t = x[:, i, :]                    # Slice X to get one time step: (B, N)
            x_step_input = x_t.unsqueeze(1)     # (B, N) -> (B, 1, N)

            # Use defualt DGCRN step
            Hidden_State, Cell_State = self.step(
                x_step_input,
                Hidden_State, 
                Cell_State,
                self.predefined_A, 
                'encoder',
                idx=None,
                i=i
            )
        
        # Pass Hidden state to Final Classification Layer
        Hidden_State_reshaped = Hidden_State.view(B, N, self.hidden_size)
        logits = self.fc_final(Hidden_State_reshaped)
        
        return logits