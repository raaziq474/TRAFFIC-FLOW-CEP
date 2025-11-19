import pandas as pd
from tsl.datasets import MetrLA
from typing import Dict, Any, List
from EventClassifier import EventClassifier
from HistoricalAverage import HistoricalAverageTraffic
from tqdm import tqdm


class CongestionDetector:

    def __init__(self, data: pd.DataFrame, min_duration: int = 3):
        """
        Initializes the detector with the speed data.
        
        Args:
            data_df: DataFrame where index is time and columns are sensor IDs (speed data).
            min_duration: Minimum number of consecutive intervals (5-minute) 
                          to qualify as a congestion event.
        """
        self.df = data
        self.min_duration = min_duration
        self.events_df: pd.DataFrame = pd.DataFrame()
        
        # Pre-process sensor IDs (0, 77123) -> 771123
        if isinstance(self.df.columns[0], tuple):
            self.df.columns = [c[0] for c in self.df.columns]

        # Train predictor once - can be reused anywhere 
        self.predictor = HistoricalAverageTraffic(self.df)
        self.predictor.train()

        
    def _add_event(self, events: List[Dict[str, Any]], sensor_id: str, start_time: pd.Timestamp, end_time: pd.Timestamp):
        """Records a congestion event using the normalized structure."""
        
        event_series = self.df.loc[start_time:end_time, sensor_id]
        duration_intervals = len(event_series)

        avg_speed = event_series.mean()     # will need to add other metrics here and pass to Event classifier
        
        day_stats = self.predictor.predict_day_distribution(start_time, sensor_id)  # getting day min and max (extracted from start event)

        classification = EventClassifier.classify_event_types(avg_speed, duration_intervals, start_time, day_stats) 
        
        if classification:

            event_data = {
                "sensor_id": sensor_id,
                "start_time": start_time,
                "end_time": end_time,
                "avg_speed": avg_speed,
                "duration_minutes": duration_intervals * 5, 
            }
            
            # Merge classifications with event data, adds fields (severity, duration_type, time_period)
            event_data.update(classification) 
            events.append(event_data)


    def detect_events(self) -> pd.DataFrame:
        """Detects events where speed is below (mean - stdev) for each sensor."""

        events = []

        pbar = tqdm(total=len(self.df.columns), desc="Detecting Events", unit="Sensor")
        for sensor_id in self.df.columns:

            is_congested = []

            # Build a boolean series for this sensor based on historical thresholds
            for timestamp, speed in zip(self.df.index, self.df[sensor_id]):

                sensor_stats = self.predictor.predict_distribution(timestamp, timestamp, sensor_id) # timestamp passd twice, change 

                sensor_mean = sensor_stats["mean"]
                sensor_stdev = sensor_stats["stdev"]

                # Congestion is defined as being slower than (mean - stdev)
                threshold = sensor_mean - sensor_stdev

                is_congested.append(speed < threshold)

            is_congested = pd.Series(is_congested, index=self.df.index)

            # Group consecutive congestion periods (True values)
            groups = (is_congested != is_congested.shift()).cumsum()
            congested_groups = self.df[is_congested].groupby(groups[is_congested])

            # Process each continuous congestion period
            for _, group in congested_groups:
                start_time = group.index[0]
                end_time = group.index[-1]
                
                if len(group) < self.min_duration:      # only add event if it is greater than minimum duration 
                    continue

                self._add_event(events, sensor_id, start_time, end_time)

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

        print("\nTime Period Distribution")
        period_summary = self.events_df.groupby('time_period').agg({
            'time_period': 'count',
            'duration_minutes': 'mean',
            'avg_speed': 'mean'
        }).rename(columns={'time_period': 'count'})
        print(period_summary)


    def run_pipeline(self):
        """Executes the full event detection and analysis pipeline."""
        self.detect_events()
        self.analyze_events()
        
        return self.events_df


if __name__ == "__main__":

    print("Downloading/Loading dataset ...")
    dataset = MetrLA(impute_zeros=True)
    df = dataset.target

    print("Dataset Loaded")
    print(f"Dataset info: {len(df.columns)} sensors, {len(df)} time steps")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    

    detector = CongestionDetector(df, min_duration=3)
    
    events_df = detector.run_pipeline()
    
    if not events_df.empty:
        events_df.to_csv("congestion_events.csv", index=False)
        print("Saved events to congestion_events.csv")