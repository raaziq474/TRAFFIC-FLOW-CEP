import pandas as pd
import numpy as np
from tsl.datasets import MetrLA
from typing import Dict, Any, List
from EventClassifier import EventClassifier
from HistoricalAverage import HistoricalAverageTraffic
from tqdm import tqdm
from CongestionChainDetector import CongestionChainDetector


class CongestionDetector:

    def __init__(self, data: pd.DataFrame, adj_matrix: np.ndarray = None, min_duration: int = 15):
        """
        Initializes the detector with the speed data.
        
        Args:
            data: DataFrame where index is time and columns are sensor IDs (speed data).
            adj_matrix: Adjacency matrix for sensor connectivity (optional, for causality detection)
            min_duration: Minimum number of consecutive intervals ( in minutes) 
                          to qualify as a congestion event.
        """
        self.df = data
        self.adj_matrix = adj_matrix
        self.min_duration = min_duration
        self.events_df: pd.DataFrame = pd.DataFrame()
        self.causal_df: pd.DataFrame = pd.DataFrame()
        
        # Pre-process sensor IDs (0, 77123) -> 771123
        if isinstance(self.df.columns[0], tuple):
            self.df.columns = [c[0] for c in self.df.columns]

        # Train predictor once - can be reused anywhere, once trained(computed)
        self.predictor = HistoricalAverageTraffic(self.df)
        self.predictor.train()

        
    def _add_event(self, events: List[Dict[str, Any]], sensor_id: str, start_time: pd.Timestamp, end_time: pd.Timestamp, event_type: str):
            """Records a congestion event using the normalized structure."""
            
            event_series = self.df.loc[start_time:end_time, sensor_id]
            duration_intervals = len(event_series)

            avg_speed = event_series.mean()                     

            day_stats = self.predictor.predict_day_distribution(start_time, sensor_id)

            # Assuming EventClassifier uses the avg_speed and day_stats to refine the classification
            classification = EventClassifier.classify_event_types(avg_speed, duration_intervals, start_time, day_stats) 
            
            if classification:

                event_data = {
                    "sensor_id": sensor_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "avg_speed": avg_speed,
                    "duration_minutes": duration_intervals * 5, 
                    "type_detected": event_type, 
                }
                
                # Merge classifications with event data
                event_data.update(classification) 
                events.append(event_data)


    def detect_events(self) -> pd.DataFrame:
        """Detects events where speed is below historical thresholds for each sensor."""

        events = []

        pbar = tqdm(total=len(self.df.columns), desc="Detecting Events", unit="Sensor")
        for sensor_id in self.df.columns:

            # is_congested now stores the type of congestion or None
            congestion_type_list = [] 

            for timestamp, speed in zip(self.df.index, self.df[sensor_id]):

                sensor_stats = self.predictor.predict_distribution(timestamp, timestamp, sensor_id)

                sensor_mean = sensor_stats["mean"]
                sensor_stdev = sensor_stats["stdev"]
                
                # 1. Random/Off-Peak Congestion (Significant)
                random_threshold = sensor_mean - sensor_stdev
                if speed < random_threshold:
                    congestion_type_list.append("random") 
                
                # 2. Peak/Random Congestion (Expected congestion)
                elif speed < 0.8 * sensor_mean:
                    congestion_type_list.append("peak") 
                
                # 3. No Congestion
                else:
                    congestion_type_list.append(None)

            # Group consecutive congestion periods (where the type is not None)
            # We use the index of the original DataFrame for grouping
            groups = (pd.Series(congestion_type_list, index=self.df.index).notna() != pd.Series(congestion_type_list, index=self.df.index).notna().shift()).cumsum()
            congested_groups = self.df[pd.Series(congestion_type_list, index=self.df.index).notna()].groupby(groups[pd.Series(congestion_type_list, index=self.df.index).notna()])

            # Process each continuous congestion period
            for group_id, group in congested_groups:
                start_time = group.index[0]
                end_time = group.index[-1]
                
                if len(group) < (self.min_duration / 5):    # Only count if event if duration longer than minimum duration
                    continue

                event_types = pd.Series(congestion_type_list, index=self.df.index).loc[start_time:end_time].dropna()
                
                # Use the most frequent event type during the congestion event
                dominant_type = event_types.mode()[0] if not event_types.empty else 'unknown' 
                self._add_event(events, sensor_id, start_time, end_time, dominant_type)

            pbar.update(1)

        pbar.close()
        self.events_df = pd.DataFrame(events)
        print(f"Detected {len(self.events_df)} congestion events.")

        return self.events_df


    def analyze_events(self):
        """Event Distribution statistics on detected events."""

        if self.events_df.empty:
            print("\nNo events to analyze.")
            return

        print("\nSeverity Distribution")
        severity_summary = self.events_df.groupby('severity').agg({
            'severity': 'count',
            'duration_minutes': 'mean',
            'avg_speed': 'mean'
        }).rename(columns={'severity': 'count'}).sort_values('count', ascending=False)
        print(severity_summary)

        print("\nEvent Type Distribution")
        period_summary = self.events_df.groupby('type_detected').agg({
            'type_detected': 'count',
            'duration_minutes': 'mean',
            'avg_speed': 'mean'
        }).rename(columns={'type_detected': 'count'})
        print(period_summary)

        print("\nDay Distribution")
        period_summary = self.events_df.groupby('day_of_week').agg({
            'day_of_week': 'count',
            'duration_minutes': 'mean',
            'avg_speed': 'mean'
        }).rename(columns={'day_of_week': 'count'})
        print(period_summary)


    def find_related_events(self, time_window_minutes: int = 15, min_connectivity: float = 0.1):
        """Finds causal relationships between detected events using adjacency matrix"""
        
        if self.events_df.empty:
            print("No events detected yet. Run detect_events() first")
            return pd.DataFrame()
        
        if self.adj_matrix is None:
            print("No adjacency matrix provided. Cannot find related events")
            return pd.DataFrame()
        
        # Get sensor IDs from dataframe columns -> adj matrix does not have sensor ids by defualt, just 0 indexed
        sensor_ids = [str(sid) for sid in self.df.columns]
        
        print(f"\nSearching for causal relationships...")
        print(f"Time window: {time_window_minutes} minutes")
        print(f"Minimum connectivity: {min_connectivity}")
        
        causal_df = CongestionChainDetector.find_causal_events(
            self.events_df,
            self.adj_matrix,
            sensor_ids,
            time_window_minutes=time_window_minutes,
            min_connectivity=min_connectivity
        )
        
        # Analyze chains: Not useful at the moment, can save to csv?
        if not causal_df.empty:
            chains = CongestionChainDetector.analyze_causal_chains(causal_df, max_depth=5)   # length of chain limited to dfs max depth
            print(f"\nFound {len(chains)} causal chains")      
            if chains:
                longest = max(chains, key=len)
                print(f"Longest chain: {len(longest)} events")
                print(f"Longest chain event IDs: {longest}")    
        
        CongestionChainDetector.summarize_causal_patterns(causal_df)
        
        self.causal_df = causal_df
        return causal_df


    def run_pipeline(self):
        """Executes the full event detection and analysis pipeline."""

        self.detect_events()
        self.analyze_events()
        self.find_related_events()

        return self.events_df


if __name__ == "__main__":

    print("Downloading/Loading dataset ...")
    dataset = MetrLA(impute_zeros=True)
    df = dataset.target
    adj = dataset.get_connectivity(method="distance", threshold=0.1, include_self=False, normalize_axis=1)

    print("Dataset and Adjacency Matrix Loaded")
    print(f"Dataset info: {len(df.columns)} sensors, {len(df)} time steps")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    detector = CongestionDetector(df, adj_matrix=adj, min_duration=15)
    events_df = detector.run_pipeline()
    
    if not events_df.empty:
        events_df.to_csv("congestion_events.csv", index=False)
        print("\nSaved events to congestion_events.csv")
    
    if not detector.causal_df.empty:
        detector.causal_df.to_csv("event_chains.csv", index=False)
        print("Saved chains events to event_chains.csv")   