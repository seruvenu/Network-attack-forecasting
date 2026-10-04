"""
ARGUS World Model Training
Temporal Transformer-based network attack forecasting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

print("[ARGUS] Initializing model training pipeline...")

# ============================================================================
# CONFIGURATION
# ============================================================================
SEQUENCE_LENGTH = 10  # 10 time steps (flows)
FORECAST_STEPS = 5    # Predict next 5 steps
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[DEVICE] Using {DEVICE}")

# ============================================================================
# SYNTHETIC DATA GENERATION (for quick demo)
# ============================================================================
def generate_synthetic_data(n_samples=5000, n_features=15):
    """
    Generate synthetic network flow data
    Features: src_port, dst_port, protocol, byte_count, packet_count, 
              duration, syn_flag, ack_flag, fin_flag, rst_flag, 
              iat_mean, iat_var, bidirectional_ratio, ttl, payload_size
    """
    print("[DATA] Generating synthetic network traffic data...")
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create attack patterns: if features have certain characteristics → attack likely
    y = np.zeros(n_samples)
    for i in range(n_samples):
        # Simple attack pattern: high SYN flags + unusual port combos + low TTL
        if X[i, 6] > 1.5 and X[i, 10] > 1.0 and X[i, 13] < -1.0:
            y[i] = 1  # Attack
        elif np.random.random() < 0.15:  # 15% baseline attack rate
            y[i] = 1
    
    return X, y

# ============================================================================
# LSTM WORLD MODEL
# ============================================================================
class WorldModelLSTM(nn.Module):
    """
    Temporal LSTM that learns P(S_t+1 | S_t)
    Takes sequence of network states, predicts future state + attack probability
    """
    def __init__(self, input_size, hidden_size, num_layers, forecast_steps):
        super(WorldModelLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_steps = forecast_steps
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention mechanism (simplified)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Decoder for multi-step forecast
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, forecast_steps)  # K-step ahead predictions
        )
        
        # Attack probability head
        self.attack_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, forecast_steps),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        Returns: attack_prob (batch_size, forecast_steps), state_pred (batch_size, forecast_steps)
        """
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # (B, T, H)
        
        # Attention over time steps
        attention_weights = self.attention(lstm_out)  # (B, T, 1)
        context = (lstm_out * attention_weights).sum(dim=1)  # (B, H)
        
        # Multi-step forecasts
        state_pred = self.decoder(context)      # (B, K)
        attack_prob = self.attack_head(context) # (B, K)
        
        return attack_prob, state_pred, attention_weights

# ============================================================================
# DATASET
# ============================================================================
class NetworkFlowDataset(Dataset):
    def __init__(self, X, y, seq_len, forecast_steps):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.forecast_steps = forecast_steps
    
    def __len__(self):
        return len(self.X) - self.seq_len - self.forecast_steps
    
    def __getitem__(self, idx):
        # Input: seq_len flows
        x_seq = self.X[idx:idx + self.seq_len]
        
        # Target: next forecast_steps attack labels
        y_seq = self.y[idx + self.seq_len:idx + self.seq_len + self.forecast_steps]
        
        return torch.FloatTensor(x_seq), torch.FloatTensor(y_seq)

# ============================================================================
# TRAINING
# ============================================================================
def train_model():
    print("\n" + "="*70)
    print("[TRAINING] Starting World Model training...")
    print("="*70)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=5000, n_features=15)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("[SCALER] Saved to scaler.pkl")
    
    # Create dataset
    dataset = NetworkFlowDataset(X_scaled, y, SEQUENCE_LENGTH, FORECAST_STEPS)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"[DATA] Total sequences: {len(dataset)}")
    print(f"[DATA] Batch size: {BATCH_SIZE}")
    print(f"[DATA] Batches per epoch: {len(train_loader)}")
    
    # Initialize model
    model = WorldModelLSTM(
        input_size=15,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        forecast_steps=FORECAST_STEPS
    ).to(DEVICE)
    
    print(f"[MODEL] Initialized WorldModelLSTM")
    print(f"[MODEL] Hidden size: {HIDDEN_SIZE}, Layers: {NUM_LAYERS}")
    
    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # Training loop
    print("\n[TRAINING] Starting epochs...\n")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            # Forward pass
            attack_prob, state_pred, attention = model(x_batch)
            
            # Loss: predict attack probability for next K steps
            loss = criterion(attack_prob, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'world_model.pt')
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"[MODEL] Best loss: {best_loss:.4f}")
    print(f"[MODEL] Saved to world_model.pt")
    
    return model, scaler

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    try:
        model, scaler = train_model()
        print("\n" + "="*70)
        print("[READY] Model trained successfully!")
        print("[NEXT] Run: python app.py")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
