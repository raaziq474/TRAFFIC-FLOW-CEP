import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tsl.datasets import MetrLA
import os
from sklearn.preprocessing import StandardScaler

from CongestionEventDataset import EventLabelGenerator, MultiSensorDataset
from models.GWNetClassifier import GWNetClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_FOLDER = "logs/gwnet"
os.makedirs("logs/gwnet", exist_ok=True)


# CONFIG
HISTORY_WINDOW = 24
FUTURE_STEPS = 6
BATCH_SIZE = 64
LR = 0.003
EPOCHS = 1

GWNET_HPARAMS = {
    'residual_channels': 32,
    'dilation_channels': 32,
    'skip_channels': 256,
    'end_channels': 512,
    'blocks': 2,
    'layers': 4,        # Default 8 layers (2 blocks * 4 layers/block)
    'dropout': 0.3,
    'kernel_size': 2 
}


# -----------------------------
# Load and Prepare Data (Same as previous steps)
# -----------------------------
label_generator = EventLabelGenerator(events_csv="data/congestion_events.csv")
speeds, labels, sensor_ids, num_classes = label_generator.run()
T, N = speeds.shape

# -----------------------------
# 70 / 15 / 15 SPLIT
# -----------------------------
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

split_1 = int(T * train_ratio)
split_2 = int(T * (train_ratio + val_ratio))

speeds_train, labels_train = speeds[:split_1], labels[:split_1]
speeds_val, labels_val     = speeds[split_1:split_2], labels[split_1:split_2]
speeds_test, labels_test   = speeds[split_2:], labels[split_2:]

# Fit scaler only on training data
scaler = StandardScaler()
scaler.fit(speeds_train.reshape(-1, 1))

speeds_train_norm = scaler.transform(speeds_train.reshape(-1, 1)).reshape(speeds_train.shape)
speeds_val_norm   = scaler.transform(speeds_val.reshape(-1, 1)).reshape(speeds_val.shape)
speeds_test_norm  = scaler.transform(speeds_test.reshape(-1, 1)).reshape(speeds_test.shape)


# MultiSensorDataset (T, N) -> (B, T, N)
dataset_train = MultiSensorDataset(speeds_train_norm, labels_train, HISTORY_WINDOW, FUTURE_STEPS)
dataset_val   = MultiSensorDataset(speeds_val_norm, labels_val, HISTORY_WINDOW, FUTURE_STEPS)
dataset_test  = MultiSensorDataset(speeds_test_norm, labels_test, HISTORY_WINDOW, FUTURE_STEPS)

loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
loader_val   = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
loader_test  = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Model Initialization
# -----------------------------

# A = dataset.compute_similarity(method="distance", threshold=0.1)  # (N, N)
# A = torch.tensor(A, dtype=torch.float32).to(device)
# supports = [A] 

model = GWNetClassifier(
    device=device,
    num_nodes=N,
    num_classes=num_classes,
    # supports=supports,
    **GWNET_HPARAMS
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

print(f"\nStarted Training GWNet Classifier:")

# -----------------------------
# Training Loop
# -----------------------------
best_val_loss = float('inf')
model_name = f"{MODELS_FOLDER}/gwnet_classifier_best_{HISTORY_WINDOW}_{FUTURE_STEPS}.pt"

for epoch in range(EPOCHS):
    # TRAIN
    model.train()
    train_losses = []
    
    for X, y in loader_train:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        out = model(X) # (B, N, num_classes)
        
        # Reshape output (B*N, num_classes) and target (B*N) for CrossEntropyLoss
        loss = loss_fn(out.reshape(-1, num_classes), y.reshape(-1))
        
        loss.backward()
        opt.step()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        train_losses.append(loss.item())
    
    train_loss = np.mean(train_losses)
    
    # VALIDATION
    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for X, y in loader_val:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = loss_fn(out.reshape(-1, num_classes), y.reshape(-1))
            val_losses.append(loss.item())
            
            preds = out.argmax(dim=2)
            val_correct += (preds == y).sum().item()
            val_total += y.numel()
    
    val_loss = np.mean(val_losses)
    val_acc = 100 * (val_correct / val_total)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_name)
    
    print(f"Epoch {epoch+1:2d}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

print(f"\nBest validation loss: {best_val_loss:.4f}")
print(f"Model saved to {model_name}")

# -----------------------------
# FINAL TEST EVALUATION
# -----------------------------
print("\n--- Running final TEST evaluation ---")
model.load_state_dict(torch.load(model_name, weights_only=False))
model.eval()

test_losses = []
test_correct = 0
test_total = 0

with torch.no_grad():
    for X, y in loader_test:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = loss_fn(out.reshape(-1, num_classes), y.reshape(-1))
        test_losses.append(loss.item())

        preds = out.argmax(dim=2)
        test_correct += (preds == y).sum().item()
        test_total += y.numel()

test_loss = np.mean(test_losses)
test_acc = 100 * test_correct / test_total

print(f"TEST Loss={test_loss:.4f} | TEST Accuracy={test_acc:.2f}%")