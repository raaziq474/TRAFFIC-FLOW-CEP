import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, recall_score
from omegaconf import OmegaConf
from hydra import initialize, compose

from models.TCNClassifier import MultiSensorTCN
from models.GWNetClassifier import GWNetClassifier
from models.D2stgnn import D2STGNN
from models.DgcrnClassifier import DGCRN_Classifier
from CongestionEventDataset import EventLabelGenerator, MultiSensorDataset
import data_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#################
# Inference module for trained traffic classification models
#################


def get_model(model_name, num_nodes, num_classes, cfg, device):
    # should probabaly return a function here instead of calling model ?
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



def main(model_path: str, dataset: str, events_dataset: str):
    """Loads model, prepares data, and runs testing."""

    # Build config
    model_dir = os.path.dirname(model_path)
    initialize(config_path=model_dir, version_base=None)
    cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))


    # Data Loading and Preparation
    print(f"Loading and Labeling Data from {dataset}...")
    df = data_utils.load_csv(dataset)
    gen = EventLabelGenerator(events_dataset, dataset=df) 
    speeds, labels, sensor_ids, num_classes = gen.run()

    T, N = speeds.shape
    print(f"Data shape speeds: {speeds.shape}, sensors: {N}, classes: {num_classes}")
    
    # Map class outputs to labels (Store Distribution)
    unique, counts = np.unique(labels, return_counts=True)
    idx_to_class = {v: k for k, v in gen.class_to_idx.items()}

    class_dist = ""
    for index, count in zip(unique, counts):
        name = idx_to_class.get(index, "UNKNOWN")
        proportion = 100 * count / labels.size
        class_dist += f"{name} (Class {index}): {count} ({proportion:.2f}%)" + '\n'

    # Partition Data
    train_end = int(T * cfg.dataset.train_split)
    val_end = int(T * (cfg.dataset.train_split + cfg.dataset.val_split))

    speeds_train = speeds[:train_end]
    speeds_test = speeds[val_end:]
    labels_test = labels[val_end:]
    
    print(f"Split: Test {len(speeds_test)} samples")

    # Normalize data
    scaler = StandardScaler()
    scaler.fit(speeds_train.reshape(-1, 1))     # Fit only on training data
    speeds_test_norm = scaler.transform(speeds_test.reshape(-1, 1)).reshape(speeds_test.shape)
    speeds_test = speeds_test_norm

    # convert to dataset to dataloader
    test_dataset = MultiSensorDataset(speeds_test, labels_test, window=cfg.window, horizon=cfg.horizon)
    loader_test = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    

    # Model Loading and Initialization
    model = get_model(cfg.model.name, N, num_classes, cfg, device)
    criterion = nn.CrossEntropyLoss(reduction="none")

    print(f"\nLoading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    best_val_loss = checkpoint.get("best_val_loss", "N/A")
    best_val_acc = checkpoint.get("best_val_accuracy", "N/A")

    test_losses = []
    all_preds = []
    all_labels = []
    all_logits = []
    
    print("\nStarting Testing...")
    with torch.no_grad():
        for X, y in loader_test:
            X = X.to(device)
            y = y.to(device)

            out = model(X)

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

    # Metrics
    test_loss = float(np.mean(test_losses))
    test_acc = accuracy_score(y_true_overall, y_pred_overall) * 100

    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Overall Results
    num_classes_report = len(idx_to_class)
    target_names = [idx_to_class[i] for i in range(num_classes_report)]

    test_results_path = os.path.join(model_dir, f"inference_results.txt")
    with open(test_results_path, "w+") as f:
        f.write("INFERENCE RESULTS")
        f.write(f"Model Directory: {model_path}\n")
        f.write(f"Validation Loss (from training): {best_val_loss:.4f}\n")
        f.write(f"Validation Accuracy (from training): {best_val_acc:.4f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}%\n\n")

        f.write(class_dist) 
        f.write("\nOverall Classification Report:\n")
        f.write(classification_report(y_true_overall, y_pred_overall, target_names=target_names, digits=3, zero_division=0))
        
        cm_overall = confusion_matrix(y_true_overall, y_pred_overall)
        f.write("\nOverall Confusion Matrix:\n")
        f.write(np.array2string(cm_overall))

    print(f"Saved overall test results to: {test_results_path}")


    # PER-SENSOR METRICS
    results = []
    criterion_sensor = nn.CrossEntropyLoss(reduction="mean")
    for sensor_idx in range(N):
        logits_s = all_logits[:, sensor_idx, :]
        y_true   = all_labels[:, sensor_idx]
        y_pred   = all_preds[:, sensor_idx]

        logits_t = torch.tensor(logits_s)
        labels_t = torch.tensor(y_true)

        sensor_loss = criterion_sensor(logits_t, labels_t).item()
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc_sensor = 100.0 * (y_pred == y_true).mean()
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

        results.append({
            "sensor_idx": sensor_idx,
            "sensor_id": sensor_ids[sensor_idx],
            "sensor_test_loss": sensor_loss,
            "sensor_test_acc": acc_sensor,
            "sensor_f1_macro": f1_macro,
            "sensor_recall_macro": recall_macro
        })

    summary_df = pd.DataFrame(results)
    results_csv = os.path.join(model_dir, "inference_sensor_performance.csv")
    summary_df.to_csv(results_csv, index=False)
    print(f"Saved per-sensor performance summary to: {results_csv}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load and test a trained spatio-temporal model on traffic congestion classification")
    parser.add_argument("model_path", type=str, 
                        help="Path to the directory containing config.yaml and the model checkpoint (e.g., logs/dgcrn-2025-12-11_16-00-00). Path must be relative")
    
    parser.add_argument("--dataset", type=str, default="la.csv",
                        help="Name of the raw traffic speeed dataset file (e.g., la.csv)")
    
    parser.add_argument("--events_dataset", type=str, default="data/congestion_events_la.csv",
                        help="Name of the events dataset CSV file (e.g., congestion_events_la.csv)")

    args = parser.parse_args()

    # Usage: python traffic_inference.py logs/dgcrn-2025-12-11_16-00-00/dgcrn.pt
    # --dataset la.csv [defualt/optional]
    # --events_dataset [optionaloptional]
    # model selscted will be from config located in that folder
    main(args.model_path, args.dataset, args.events_dataset)