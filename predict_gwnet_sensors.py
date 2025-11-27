import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tsl.datasets import MetrLA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os
from models.GWNetClassifier import GWNetClassifier
from CongestionEventDataset import EventLabelGenerator


from CongestionEventDataset import MultiSensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_per_sensor(model, loader, sensor_ids, num_classes, device):
    """Evaluate model performance for each sensor individually"""

    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction='none')  # Get per-sample losses
    
    N_SENSORS = len(sensor_ids)
    
    sensor_results = {i: {'losses': [], 'correct': 0, 'total': 0} for i in range(N_SENSORS)}
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)  # (B, N, num_classes)
            
            BATCH_SIZE = X.shape[0]
            preds = out.argmax(dim=2)
            
            # Calculate loss for each sample and sensor
            out_flat = out.reshape(BATCH_SIZE * N_SENSORS, num_classes)
            y_flat = y.reshape(BATCH_SIZE * N_SENSORS)
            losses = loss_fn(out_flat, y_flat)
            losses = losses.reshape(BATCH_SIZE, N_SENSORS)
            
            # Accumulate statistics for each sensor
            for sensor_idx in range(N_SENSORS):
                sensor_losses = losses[:, sensor_idx].cpu().numpy()
                sensor_preds = preds[:, sensor_idx]
                sensor_targets = y[:, sensor_idx]
                
                sensor_results[sensor_idx]['losses'].extend(sensor_losses.tolist())
                sensor_results[sensor_idx]['correct'] += (sensor_preds == sensor_targets).sum().item()
                sensor_results[sensor_idx]['total'] += BATCH_SIZE
    
    return sensor_results


def compute_sensor_metrics(sensor_results):
    """Compute average loss and accuracy for each sensor"""

    metrics = {}
    for sensor_idx, data in sensor_results.items():
        avg_loss = np.mean(data['losses'])
        accuracy = 100.0 * data['correct'] / data['total']
        metrics[sensor_idx] = {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    return metrics


def compute_classification_report(model, loader, idx_to_class, device):
    """
    Computes and returns the scikit-learn classification report (precision, recall, f1-score)
    for all classes across all sensors and samples in the loader.
    """
    model.eval()
    
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            
            # model output: (B, N, num_classes)
            out = model(X)
            preds = out.argmax(dim=2)

            # flatten everything into shape (B*N)
            y_flat = y.reshape(-1).cpu().numpy()
            preds_flat = preds.reshape(-1).cpu().numpy()

            all_targets.append(y_flat)
            all_predictions.append(preds_flat)

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    # Convert the class ID mapping back to class names for a readable report
    # Ensure target names are ordered by class ID
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    report = classification_report(
        all_targets, 
        all_predictions, 
        target_names=target_names, 
        zero_division=0,
        output_dict=False
    )

    return report


if __name__ == '__main__':

    # Load dataset and prepare data
    label_generator = EventLabelGenerator(events_csv="data/congestion_events.csv")
    speeds, labels, sensor_ids, num_classes = label_generator.run()
    T, N = speeds.shape

    # Map node index to class labels (eg. 5 -> no_event)
    idx_to_class = {i: c for c, i in label_generator.class_to_idx.items()}

    # Train/Val/Test split
    train_ratio = 0.70
    val_ratio = 0.15
    split_1 = int(T * train_ratio)
    split_2 = int(T * (train_ratio + val_ratio))

    speeds_val = speeds[split_1:split_2]
    labels_val = labels[split_1:split_2]
    speeds_test = speeds[split_2:]
    labels_test = labels[split_2:]

    # Normalize
    speeds_train = speeds[:split_1]
    scaler = StandardScaler()
    scaler.fit(speeds_train.reshape(-1, 1))

    speeds_val_norm = scaler.transform(speeds_val.reshape(-1, 1)).reshape(speeds_val.shape)
    speeds_test_norm = scaler.transform(speeds_test.reshape(-1, 1)).reshape(speeds_test.shape)

    # Create datasets and loaders
    HISTORY_WINDOW = 12
    FUTURE_STEPS = 12
    BATCH_SIZE = 64

    dataset_val = MultiSensorDataset(speeds_val_norm, labels_val, HISTORY_WINDOW, FUTURE_STEPS)
    dataset_test = MultiSensorDataset(speeds_test_norm, labels_test, HISTORY_WINDOW, FUTURE_STEPS)

    loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    # Load the trained model
    GWNET_HPARAMS = {
        'residual_channels': 32,
        'dilation_channels': 32,
        'skip_channels': 256,
        'end_channels': 512,
        'blocks': 2,
        'layers': 4,
        'dropout': 0.3,
        'kernel_size': 2
    }

    model = GWNetClassifier(
        device=device,
        num_nodes=N,
        num_classes=num_classes,
        **GWNET_HPARAMS
    ).to(device)

    model_path = "logs/gwnet/gwnet_classifier_best_12_12.pt"  # Model to generate per sensor loss/accuracy
    
    # Check if model file exists before loading
    if not os.path.exists(model_path):
        raise Exception(f"\nERROR: Model file not found at '{model_path}'")

    model.load_state_dict(torch.load(model_path, map_location=device))

    print("Evaluating validation set...")
    val_sensor_results = evaluate_per_sensor(model, loader_val, sensor_ids, num_classes, device)
    val_metrics = compute_sensor_metrics(val_sensor_results)

    print("Evaluating test set...")
    test_sensor_results = evaluate_per_sensor(model, loader_test, sensor_ids, num_classes, device)
    test_metrics = compute_sensor_metrics(test_sensor_results)

    # Compile results into a DataFrame
    results_list = []
    for sensor_idx in range(len(sensor_ids)):
        results_list.append({
            'sensor_idx': sensor_idx,
            'sensor_id': sensor_ids[sensor_idx],
            'val_loss': val_metrics[sensor_idx]['loss'],
            'test_loss': test_metrics[sensor_idx]['loss'],
            'test_acc': test_metrics[sensor_idx]['accuracy']
        })

    results_df = pd.DataFrame(results_list)

    # Save to CSV
    output_path = f'results/sensor_performance_gwnet_{HISTORY_WINDOW}_{FUTURE_STEPS}.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'\n")

    print(f"Mean Val Loss: {results_df['val_loss'].mean():.4f}")
    print(f"Mean Test Loss: {results_df['test_loss'].mean():.4f}")
    print(f"Mean Test Accuracy: {results_df['test_acc'].mean():.2f}%")

    classification_report_output = compute_classification_report(
        model, 
        loader_test, 
        idx_to_class, 
        device
    )

    print(classification_report_output)