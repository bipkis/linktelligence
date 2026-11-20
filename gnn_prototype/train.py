import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from dataset import NetworkDataset
from model import NetworkGNN
import os

# --- Configuration ---
DATA_ROOT = "." # Assuming data/ directory is in the current working directory
TRAIN_SPLIT = 0.8
EPOCHS = 50
BATCH_SIZE = 32
MODEL_PATH = "best_model.pt"

def train():
    """Loads dataset, trains the GNN model, and saves weights."""
    print("--- Starting Model Training ---")
    
    # 1. Load Dataset
    try:
        dataset = NetworkDataset(root=DATA_ROOT)
        if len(dataset) == 0:
            print("Error: No data found in the 'data/' directory. Please run generate_data.py first.")
            return

        print(f"Successfully loaded {len(dataset)} samples from the dataset.")
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Split Data (Simple split for demonstration)
    train_size = int(TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 3. Initialize Model and Loss/Optimizer
    # In-channels: Node features (7 types). Hidden channels: 64. Out_channels is handled by heads.
    model = NetworkGNN(in_channels=7, hidden_channels=64, out_channels=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Loss function initialization
    score_criterion = nn.MSELoss() # For continuous score prediction [0, 1]
    bottleneck_criterion = nn.CrossEntropyLoss() # For classification flag (CE loss expects class indices)
    security_criterion = nn.CrossEntropyLoss() # For classification flag

    print("Starting training loop...")
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_epoch_loss = 0
        
        # Training Loop
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        for data in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            score, bottleneck, security_flaw = model(data.x, data.edge_index, data.edge_attr)
            
            # Calculate loss for all three heads using appropriate criteria
            # Score (MSE) - Target Index 0
            loss_score = score_criterion(score, data.y[:, 0].unsqueeze(-1))
            # Bottleneck (CE Loss expects class indices as long tensor) - Target Index 1
            loss_bottleneck = bottleneck_criterion(bottleneck, data.y[:, 1].long())
            # Security Flaw (CE Loss expects class indices as long tensor) - Target Index 2
            loss_security = security_criterion(security_flaw, data.y[:, 2].long())
            
            total_loss = loss_score + loss_bottleneck + loss_security
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            total_epoch_loss += total_loss.item()

        avg_train_loss = total_epoch_loss / len(train_loader)

        # --- Evaluation ---
        model.eval()
        test_loss = 0
        with torch.no_grad():
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            for data in test_loader:
                score, bottleneck, security_flaw = model(data.x, data.edge_index, data.edge_attr)
                loss_score = score_criterion(score, data.y[:, 0].unsqueeze(-1))
                loss_bottleneck = bottleneck_criterion(bottleneck, data.y[:, 1].long())
                loss_security = security_criterion(security_flaw, data.y[:, 2].long())
                test_loss += (loss_score + loss_bottleneck + loss_security).item()

        avg_test_loss = test_loss / len(test_loader)


        print(f'Epoch: {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}')

        # Save the best model based on test loss
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH} (New Best Loss)")

    print("--- Training Complete ---")


if __name__ == "__main__":
    train()