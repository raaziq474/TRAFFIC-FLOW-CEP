import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime

from CongestionEventDataset import EventLabelGenerator, MultiSensorDataset
from models.TCNClassifier import MultiSensorTCN
from models.GWNetClassifier import GWNetClassifier
from models.D2stgnn import D2STGNN
from models.DgcrnClassifier import DGCRN_Classifier
import data_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name, num_nodes, num_classes, cfg, device):

    if model_name == 'tcn':
        return MultiSensorTCN(num_nodes=num_nodes, num_classes=num_classes, **cfg.model.hparams).to(device)
    elif model_name == 'gwnet':
        return GWNetClassifier(device=device, num_nodes=num_nodes, num_classes=num_classes, **cfg.model.hparams).to(device)
    elif model_name == 'dgcrn':
        return DGCRN_Classifier(num_nodes=num_nodes, num_classes=num_classes, device=device, **cfg.model.hparams).to(device)
    elif model_name == 'd2stgnn':   # Not Implemented Properly !!!
        return D2STGNN(num_nodes=num_nodes, num_classes=num_classes).to(device)
    else:
        raise NotImplementedError(f"Model Not Implemented: {model_name}")


# build configuration tree from yaml config 
config_dir = os.path.dirname("config/")
initialize(config_path=config_dir, version_base=None)
cfg = compose(config_name="default")
print(OmegaConf.to_yaml(cfg))

dataset = cfg.dataset
model_name = cfg.model.name


print(f"Generating Labelled Dataset...")
df = data_utils.load_csv("la.csv")                 # defualt path is data
gen = EventLabelGenerator("data/congestion_events_la.csv", dataset=df)
speeds, labels, sensor_ids, num_classes = gen.run()

T, N = speeds.shape
print(f"Data shape speeds: {speeds.shape}, labels: {labels.shape}, sensors: {N}, classes: {num_classes}")

# Overall class distribution
idx_to_class = {v: k for k, v in gen.class_to_idx.items()}
unique, counts = np.unique(labels, return_counts=True)

class_dist = ""
for index, count in zip(unique, counts):
    name = idx_to_class.get(index, "UNKNOWN")
    proportion = 100 * count / labels.size
    class_dist += f"{name} (Class {index}): {count} ({proportion:.2f}%)" + '\n'
print(class_dist)


# Train/Val/Test split along time dimension
train_end = int(T * cfg.dataset.train_split)
val_end = int(T * (cfg.dataset.train_split + cfg.dataset.val_split))

speeds_train = speeds[:train_end]
labels_train = labels[:train_end]

speeds_val = speeds[train_end:val_end]
labels_val = labels[train_end:val_end]

speeds_test = speeds[val_end:]
labels_test = labels[val_end:]

print(f"\nSplit: Train {len(speeds_train)}, Val {len(speeds_val)}, Test {len(speeds_test)}")


# Normalize data 
scaler = StandardScaler()
scaler.fit(speeds_train.reshape(-1, 1))

speeds_train_norm = scaler.transform(speeds_train.reshape(-1, 1)).reshape(speeds_train.shape)
speeds_val_norm = scaler.transform(speeds_val.reshape(-1, 1)).reshape(speeds_val.shape)
speeds_test_norm = scaler.transform(speeds_test.reshape(-1, 1)).reshape(speeds_test.shape)

speeds_train = speeds_train_norm
speeds_val = speeds_val_norm
speeds_test = speeds_test_norm


# DATASETS AND LOADERS
train_dataset = MultiSensorDataset(speeds_train, labels_train, window=cfg.window , horizon=cfg.horizon)
val_dataset = MultiSensorDataset(speeds_val, labels_val, window=cfg.window , horizon=cfg.horizon)
test_dataset = MultiSensorDataset(speeds_test, labels_test, window=cfg.window , horizon=cfg.horizon)

loader_train = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
loader_val = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
loader_test = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)


# MODEL INITIALIZATION
model = get_model(model_name=model_name, num_nodes=N, num_classes=num_classes, cfg=cfg, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.hparams.lr)
criterion = nn.CrossEntropyLoss()

best_val_loss = float("inf")
patience_counter = 0

model_path = f"logs/{model_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
best_model_path = f"{model_path}/{model_name}.pt"
os.makedirs(model_path, exist_ok=True)

def epoch_eval(model, loader, device):
    """Helper function for validation loop"""
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)  # (B,H,N)
            y = y.to(device)  # (B,N)
            out = model(X)    # (B,N,C)
            loss = criterion(out.view(-1, num_classes), y.view(-1))
            losses.append(loss.item())

            preds = out.argmax(dim=2)  # (B,N)
            correct += (preds == y).sum().item()
            total += y.numel()
            
    mean_loss = np.mean(losses) if losses else 0.0
    acc = 100.0 * (correct / total) if total > 0 else 0.0
    return mean_loss, acc


# TRAINING LOOP
print("\nStarting training...")
for epoch in range(1, cfg.epochs + 1):
    model.train()
    train_losses = []
    train_correct = 0
    train_total = 0

    for X, y in loader_train:
        X = X.to(device)  # (B,H,N)
        y = y.to(device)  # (B,N)

        optimizer.zero_grad()
        out = model(X)    # (B,N,C)

        loss = criterion(out.view(-1, num_classes), y.view(-1))
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_val)
        
        optimizer.step()

        train_losses.append(loss.item())
        preds = out.argmax(dim=2)
        train_correct += (preds == y).sum().item()
        train_total += y.numel()

    train_loss = np.mean(train_losses) if train_losses else 0.0
    train_acc = 100.0 * (train_correct / train_total) if train_total > 0 else 0.0

    val_loss, val_acc = epoch_eval(model, loader_val, device)

    print(f"Epoch {epoch:02d} | Train Loss {train_loss:.4f}, Acc {train_acc:.2f}% | Val Loss {val_loss:.4f}, Acc {val_acc:.2f}%")

    # Early stopping & save best
    if val_loss < best_val_loss - 1e-6:  #  1e-6 ensures loss is actually lower not by some floating error 
        best_val_loss = val_loss
        best_val_acc = val_acc
        patience_counter = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_acc,
        }, best_model_path)
        print(f"-- Saved best model (val_loss improved to {val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement. patience {patience_counter}/{cfg.patience}")

    if patience_counter >= cfg.patience:
        print("Early stopping triggered.")
        break


# TESTING
print("\nLoading best model for testing...")
checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss(reduction="none")

test_losses = []
all_preds = []
all_labels = []
all_logits = []

with torch.no_grad():
    for X, y in loader_test:
        X = X.to(device)
        y = y.to(device)

        out = model(X)

        # Raw logits extracted - needed for per sensor loss 
        all_logits.append(out.cpu().numpy())
        all_labels.append(y.cpu().numpy())

        # Overall loss
        loss = criterion(out.view(-1, num_classes), y.view(-1))
        test_losses.append(loss.mean().item())

        # Predictions
        preds = out.argmax(dim=2)
        all_preds.append(preds.cpu().numpy())

# Stack everything
all_logits = np.concatenate(all_logits, axis=0)
all_preds  = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

y_true_overall = all_labels.reshape(-1)
y_pred_overall = all_preds.reshape(-1)

# OVERALL METRICS
test_loss = float(np.mean(test_losses))
test_acc = accuracy_score(y_true_overall, y_pred_overall) * 100

best_val_loss = checkpoint["best_val_loss"]
best_val_acc = checkpoint["best_val_accuracy"]

# labels for f1 matrix
num_classes = len(idx_to_class)
target_names = [idx_to_class[i] for i in range(num_classes)]

test_results = os.path.join(model_path, f"test_f1.txt")
with open(test_results, "w") as f:

    f.write("OVERALL RESULTS\n")
    f.write(f"Val Loss: {best_val_loss:.4f}\n")
    f.write(f"Val Acc: {best_val_acc:.4f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Acc: {test_acc}%\n\n")

    f.write(class_dist) 

    f.write("\nOverall Classification Report:\n")
    f.write(classification_report(y_true_overall, y_pred_overall, target_names=target_names, digits=3, zero_division=0))
    
    cm_overall = confusion_matrix(y_true_overall, y_pred_overall)
    f.write("\nOverall Confusion Matrix:\n")
    f.write(np.array2string(cm_overall))

print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc}%")

# PER-SENSOR METRICS
results = []
criterion_sensor = nn.CrossEntropyLoss(reduction="mean")
for sensor_idx in range(N):

    logits_s = all_logits[:, sensor_idx, :]
    y_true   = all_labels[:, sensor_idx]
    y_pred   = all_preds[:, sensor_idx]

    # Convert to tensors for CE loss
    logits_t = torch.tensor(logits_s)
    labels_t = torch.tensor(y_true)

    # Metrics
    sensor_loss = criterion_sensor(logits_t, labels_t).item()
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc_sensor = 100.0 * (y_pred == y_true).mean()

    results.append({
        "sensor_idx": sensor_idx,
        "sensor_id": sensor_ids[sensor_idx],
        "sensor_test_loss": sensor_loss,
        "sensor_test_acc": acc_sensor,
        "sensor_f1_macro": f1_macro
    })


# Save sensor results to csv 
summary_df = pd.DataFrame(results)
results_csv = os.path.join(model_path, f"sensor_performance_{model_name}_{cfg.window}_{cfg.horizon}.csv")
summary_df.to_csv(results_csv, index=False)
print(f"\nSaved sensor performance summary to: {results_csv}")

# Save config
config_save_path = os.path.join(model_path, "config.yaml")
with open(config_save_path, "w") as f:
    f.write(OmegaConf.to_yaml(cfg))