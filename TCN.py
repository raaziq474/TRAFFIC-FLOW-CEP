import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tsl.datasets import MetrLA
import os

from models.TCNClassifier import TCN
from CongestionEventDataset import EventLabelGenerator, SingleSensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("logs/tcn", exist_ok=True)

gen = EventLabelGenerator("data/congestion_events.csv")
speeds, labels, sensor_ids, num_classes = gen.run()

# Calculate overall class distribution
unique, counts = np.unique(labels, return_counts=True)
print("\nOverall class distribution:")
for u, c in zip(unique, counts):
    pct = 100 * c / labels.size
    print(f"  Class {u}: {c} ({pct:.2f}%)")


# --- CONFIGURATION ---
history_window = 24       # 6 hours
future_steps = 48
batch_size = 64
num_epochs = 20
learning_rate = 0.003
patience = 5               # Early stopping


# --- TRAIN / VAL / TEST SPLIT ---
train_ratio = 0.7
val_ratio   = 0.15
test_ratio  = 0.15

T, N = speeds.shape

train_end = int(T * train_ratio)
val_end   = int(T * (train_ratio + val_ratio))

speeds_train = speeds[:train_end]
labels_train = labels[:train_end]

speeds_val = speeds[train_end:val_end]
labels_val = labels[train_end:val_end]

speeds_test = speeds[val_end:]
labels_test = labels[val_end:]

print(f"Train: {len(speeds_train)}, Val: {len(speeds_val)}, Test: {len(speeds_test)}")


results = []
# TRAIN ONE TCN MODEL PER SENSOR
for sensor_idx in range(N):

    sensor_id = sensor_ids[sensor_idx]

    dataset_train = SingleSensorDataset(speeds_train, labels_train, sensor_idx, history_window, future_steps)
    dataset_val = SingleSensorDataset(speeds_val, labels_val, sensor_idx, history_window, future_steps)
    dataset_test = SingleSensorDataset(speeds_test, labels_test, sensor_idx, history_window, future_steps)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # model + opt
    model = TCN(num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):

        # TRAIN
        model.train()
        train_losses = []
        correct = 0; total = 0

        for X, y in loader_train:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)

            loss.backward()
            opt.step()

            train_losses.append(loss.item())
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()

        train_acc = 100 * correct / total
        train_loss = np.mean(train_losses)

        # VALIDATION
        model.eval()
        val_losses = []
        correct = 0; total = 0

        with torch.no_grad():
            for X, y in loader_val:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = loss_fn(out, y)

                val_losses.append(loss.item())
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()

        val_acc = 100 * correct / total
        val_loss = np.mean(val_losses)

        print(f"Sensor {sensor_idx} | Epoch {epoch+1:02d} | "
              f"Train Loss {train_loss:.4f}, Acc {train_acc:.2f}% | "
              f"Val Loss {val_loss:.4f}, Acc {val_acc:.2f}%")

        # EARLY STOPPING
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"logs/tcn/sensor_{sensor_idx}.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    # TESTING 
    model.load_state_dict(torch.load(f"logs/tcn/sensor_{sensor_idx}.pt", weights_only=False))
    model.eval()

    correct = 0; total = 0
    test_losses = []

    with torch.no_grad():
        for X, y in loader_test:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = loss_fn(out, y)
            test_losses.append(loss.item())
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()

    test_acc = 100 * (correct / total)  
    test_loss = np.mean(test_losses)

    print(f"Sensor {sensor_idx} TEST → Loss {test_loss:.4f}, Acc {test_acc:.2f}%")

    results.append({
        "sensor_idx": sensor_idx,
        "sensor_id": sensor_id,
        "val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc
    })


# SAVE RESULTS
summary_df = pd.DataFrame(results)
summary_df.to_csv(f"results/sensor_performance_tcn_{history_window}_{future_steps}.csv", index=False)
print("\nSaved sensor performance summary!")
