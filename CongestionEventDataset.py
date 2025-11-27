import numpy as np
import pandas as pd
from tsl.datasets import MetrLA
from torch.utils.data import Dataset
import torch


class EventLabelGenerator:
    def __init__(self, events_csv, dataset=None):
        """
        events_csv : path to congestion events file
        dataset    : TSL/Datetime dataset instance (defualt=METR-LA)
        """

        self.events_csv = events_csv
        self.dataset = dataset

        self.timestamps = None
        self.sensor_ids = None
        self.speeds = None
        self.TIMESTEPS = None
        self.N_SENSORS = None

        self.events = None
        self.class_to_idx = None
        self.num_classes = None
        self.labels = None
        self.sensor_id_to_idx = None


    def load_dataset(self):
        """Load dataset, and create sensor mappings sensor mappings """

        # If user did not pass in a dataset, load METR-LA
        if self.dataset is None:
            self.dataset = MetrLA(impute_zeros=True)

        df = self.dataset.target

        # Pre-process sensor IDs (0, 77123) -> 771123 
        if isinstance(df.columns[0], tuple):
            df.columns = [c[0] for c in df.columns]

        self.timestamps = df.index.to_numpy()
        self.sensor_ids = df.columns.tolist()
        self.speeds = df.to_numpy()
        self.TIMESTEPS, self.N_SENSORS = self.speeds.shape

        # Map sensor IDs to column indices
        self.sensor_id_to_idx = {str(sid): idx for idx, sid in enumerate(self.sensor_ids)}


    def load_events(self):
        """Load congestion-style events and build class mapping."""

        events = pd.read_csv(self.events_csv, parse_dates=["start_time", "end_time"])

        # Create labels for each event combination  
        events["class_label"] = events["type_detected"] + "_" + events["severity"]

        # Map event classes
        self.class_to_idx = {c: i for i, c in enumerate(events["class_label"].unique())}

        no_event_label = "no_event"
        no_event_id = len(self.class_to_idx)
        self.class_to_idx[no_event_label] = no_event_id

        events["label_id"] = events["class_label"].map(self.class_to_idx)

        self.events = events
        self.num_classes = len(self.class_to_idx)


    def generate_labels(self):
        """Create (T, N) label matrix from events."""

        no_event_id = self.class_to_idx["no_event"]

        # Initialize full label matrix
        self.labels = no_event_id * np.ones((self.TIMESTEPS, self.N_SENSORS), dtype=np.int32)

        for _, row in self.events.iterrows():

            sid = row["sensor_id"]
            sid_str = str(sid)

            if sid_str not in self.sensor_id_to_idx:
                continue

            n = self.sensor_id_to_idx[sid_str]

            start, end = row["start_time"], row["end_time"]
            label = row["label_id"]

            # Boolean mask for timestamps in window:
            #   -- for duration of event at sensor, sensor is classified as that event 
            mask = (self.timestamps >= start) & (self.timestamps <= end)

            if mask.any():
                self.labels[mask, n] = label

        return self.labels

    def run(self):
       
        self.load_dataset()
        self.load_events()

        return self.speeds, self.generate_labels(), self.sensor_ids, self.num_classes


class MultiSensorDataset(Dataset):
    
    def __init__(self, speeds, labels, history_window=12, future_steps=3):
        self.speeds = speeds
        self.labels = labels
        self.history_window = history_window
        self.future_steps = future_steps
        self.T, self.N = speeds.shape
        self.valid_idx = np.arange(history_window - 1, self.T - future_steps)
    
    def __len__(self):
        return len(self.valid_idx)
    
    def __getitem__(self, idx):
        t = self.valid_idx[idx]
        x = self.speeds[t - self.history_window + 1 : t + 1] 
        y = self.labels[t + self.future_steps]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
    
class SingleSensorDataset(Dataset):

    def __init__(self, speeds, labels, sensor_index, history_window=12, future_steps=3):

        self.x = speeds[:, sensor_index]
        self.y = labels[:, sensor_index]
        self.history_window = history_window
        self.future_steps = future_steps

        T_total = len(self.y)
        self.valid_idx = np.arange(history_window - 1, T_total - future_steps)

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        t = self.valid_idx[idx]
        x = self.x[t - self.history_window + 1 : t + 1]
        y = self.y[t + self.future_steps]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)