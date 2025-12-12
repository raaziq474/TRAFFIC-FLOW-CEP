import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import data_utils


class EventLabelGenerator:

    def __init__(self, events_csv, dataset=None):
        """
        events_csv : path to congestion events file
        dataset : TSL/Datetime dataset instance (default=METR-LA)
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

        # If user did not pass in a dataset, load METR-LA
        if self.dataset is None:
            self.dataset = data_utils.load_csv("la")

        df = self.dataset
        
        self.timestamps = df.index.to_numpy()
        self.sensor_ids = df.columns.tolist()
        self.speeds = df.to_numpy()
        self.TIMESTEPS, self.N_SENSORS = self.speeds.shape
        
        # Map sensor IDs to column indices
        self.sensor_id_to_idx = {str(sid): idx for idx, sid in enumerate(self.sensor_ids)}

    def load_events(self):
        """Load congestion events and build class mapping between labels and indices"""

        events = pd.read_csv(self.events_csv, parse_dates=["start_time", "end_time"])
        
        event_types = "type"  # (type | type_intensity)
        
        if event_types == "type_intensity":
            events["class_label"] = events["type_detected"] + "_" + events["severity"]

        elif event_types == "type":
            events["class_label"] = events["type_detected"]
        
        # can add other combinations 

        # Map event classes
        self.class_to_idx = {c: i for i, c in enumerate(events["class_label"].unique())}
        no_event_label = "no_event"
        no_event_id = len(self.class_to_idx)

        self.class_to_idx[no_event_label] = no_event_id
        events["label_id"] = events["class_label"].map(self.class_to_idx)
        
        # Filter out events for sensors not in dataset
        events["sensor_id_str"] = events["sensor_id"].astype(str)
        events = events[events["sensor_id_str"].isin(self.sensor_id_to_idx)].copy()
        
        # Map sensor IDs to indices immediately
        events["sensor_idx"] = events["sensor_id_str"].map(self.sensor_id_to_idx)
        
        self.events = events
        self.num_classes = len(self.class_to_idx)

    # def generate_labels(self):
    #     """Create (T, N) label matrix from events - OLD VERSION (SLOW)."""
    #     no_event_id = self.class_to_idx["no_event"]
        
    #     # Initialize full label matrix
    #     self.labels = no_event_id * np.ones((self.TIMESTEPS, self.N_SENSORS), dtype=np.int32)
        
    #     timestamps_dt = pd.to_datetime(self.timestamps)
        
    #     # Group events by sensor for batch processing
    #     for sensor_idx, group in self.events.groupby("sensor_idx"):
    #         # Use searchsorted for efficient time range queries
    #         for _, row in group.iterrows():

    #             start_idx = np.searchsorted(timestamps_dt, row["start_time"], side='left')
    #             end_idx = np.searchsorted(timestamps_dt, row["end_time"], side='right')
                
    #             if start_idx < end_idx:
    #                 self.labels[start_idx:end_idx, sensor_idx] = row["label_id"]
        
    #     return self.labels
    
    def generate_labels(self):
        """ Create (T, N) label matrix from events """

        no_event_id = self.class_to_idx["no_event"]
        self.labels = no_event_id * np.ones((self.TIMESTEPS, self.N_SENSORS), dtype=np.int32)
        
        if len(self.events) == 0:
            return self.labels
        
        # Convert timestamps to datetime64 (needed for searchsorted)
        timestamps_dt = pd.to_datetime(self.timestamps)
        
        # Calculate all start/end indices for all events
        start_indices = np.searchsorted(
            timestamps_dt.values, 
            self.events["start_time"].values.astype(timestamps_dt.dtype), 
            side='left'
        )
        end_indices = np.searchsorted(
            timestamps_dt.values, 
            self.events["end_time"].values.astype(timestamps_dt.dtype), 
            side='right'
        )

        # Filter out events where the time range is invalid (start_idx >= end_idx)
        valid_mask = start_indices < end_indices
        
        valid_events = self.events[valid_mask]
        start_indices_valid = start_indices[valid_mask]
        end_indices_valid = end_indices[valid_mask]
        
        sensor_indices = valid_events["sensor_idx"].values
        label_ids = valid_events["label_id"].values
        
        # Iterate through the indices and values and assign the label_id to the slice [start_idx:end_idx, sensor_idx]
        for start_idx, end_idx, sensor_idx, label_id in zip(
            start_indices_valid, end_indices_valid, sensor_indices, label_ids):

            self.labels[start_idx:end_idx, sensor_idx] = label_id
            
        return self.labels
    

    def run(self):
        """Run the full pipeline of loading raw dataset, events dataset and generating labelled pairs"""
        self.load_dataset()
        self.load_events()
        
        labels = self.generate_labels()
        
        return self.speeds, labels, self.sensor_ids, self.num_classes


class MultiSensorDataset(Dataset):
    """Dataset that maps a sequnce of inputs of all sensors to a label at some future timestep"""

    def __init__(self, speeds, labels, window=12, horizon=3):
        self.speeds = speeds
        self.labels = labels
        self.window = window
        self.horizon = horizon
        self.T, self.N = speeds.shape
        self.valid_idx = np.arange(window - 1, self.T - horizon)
    
    def __len__(self):
        return len(self.valid_idx)
    
    def __getitem__(self, idx):
        t = self.valid_idx[idx]
        x = self.speeds[t - self.window + 1 : t + 1] 
        y = self.labels[t + self.horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
    
class SingleSensorDataset(Dataset):
    """Dataset that maps a sequnce of inputs of a single sensors to a label at some future timestep"""

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
    

class MultiHorizonDataset(Dataset):
    """Dataset that maps a sequence of inputs  to a sequence of labels for all sensors"""

    def __init__(self, speeds, labels, window=144, horizon=3):
        self.speeds = speeds
        self.labels = labels
        self.window = window
        self.horizon = horizon
        self.T, self.N = speeds.shape
        
        self.valid_idx = np.arange(window - 1, self.T - horizon)
    
    def __len__(self):
        return len(self.valid_idx)
    
    def __getitem__(self, idx):

        # 't' is the last time step in the history window
        t = self.valid_idx[idx]
        x = self.speeds[t - self.window + 1 : t + 1] 
        
        # Output Y: Sequence of 'horizon' labels (B, Horizon, N)
        y = self.labels[t + 1 : t + 1 + self.horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)