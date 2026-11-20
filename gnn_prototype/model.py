import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Optional

class NetworkGNN(nn.Module):
    """
    Graph Neural Network model for predicting network topology labels.
    Uses Graph Attention Networks (GAT) for feature extraction.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(NetworkGNN, self).__init__()
        
        # GATConv layer definition
        # We use 2 attention heads (heads=2) as is common practice.
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4)

        # Global pooling layer (no weights needed here)
        self.global_pool = global_mean_pool

        # Output Heads: The pooled feature dimension is 'hidden_channels'
        # Note: We use a linear layer for the final output mapping from the pooled features.
        
        # 1. Score Head (Regression, 0 to 1)
        self.head_score = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

        # 2. Bottleneck Head (Binary Classification: 0 or 1)
        self.head_bottleneck = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 2) # Outputting 2 values for CE loss
        )

        # 3. Security Flaw Head (Binary Classification: 0 or 1)
        self.head_security = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 2) # Outputting 2 values for CE loss
        )

    @classmethod
    def from_pretrained(cls, model_path: str):
        """Loads the model state dictionary."""
        model = cls(in_channels=7, hidden_channels=64, out_channels=3) # Dummy init values
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, edge_attr: Optional[torch.Tensor] = None):
        # 1. Graph Convolution Layers
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)

        # 2. Global Pooling (Aggregate node features into a graph-level feature vector)
        # batch=None because we process one graph at a time in this setup
        x = self.global_pool(x, num_nodes=x.size(0))

        # 3. Output Heads
        score = self.head_score(x)
        bottleneck = self.head_bottleneck(x)
        security_flaw = self.head_security(x)
        
        return score, bottleneck, security_flaw

if __name__ == '__main__':
    # Example usage (requires dummy data setup)
    print("Model defined successfully.")