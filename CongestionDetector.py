import pandas as pd
from tsl.datasets import MetrLA
from typing import Dict, Any, List
from EventClassifier import EventClassifier


class CongestionDetector:

    def __init__(self, data_df: pd.DataFrame, min_duration: int = 3):
        """
        Initializes the detector with the speed data.
        
        Args:
            data_df: DataFrame where index is time and columns are sensor IDs (speed data).
            min_duration: Minimum number of consecutive intervals (5-minute) 
                          to qualify as a congestion event.
        """
        self.df = data_df
        self.min_duration = min_duration
        self.events_df: pd.DataFrame = pd.DataFrame()
        
        # Pre-process sensor IDs (0, 77123) -> 771123
        if isinstance(self.df.columns[0], tuple):
            self.df.columns = [c[0] for c in self.df.columns]

        
    def _add_event(self, events: List[Dict[str, Any]], sensor_id: str, 
                start_time: pd.Timestamp, end_time: pd.Timestamp):
        """Records a congestion event using the normalized structure."""
        
        event_series = self.df.loc[start_time:end_time, sensor_id]
        duration_intervals = len(event_series)
        
        if duration_intervals < self.min_duration:
            return

        # will need to add other metrics here and pass to Event classifier
        avg_speed = event_series.mean()
        
        classification = EventClassifier.classify_event_types(avg_speed, duration_intervals, start_time) 
        
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
        """Detects consecutive time intervals where speed is below 65 mph."""
        events = []

        # Iterate over each sensor
        for sensor_id in self.df.columns:
            
            # Probaby need to change 
            is_congested = self.df[sensor_id] < 65
            
            # Group consecutive True values (congested periods)
            groups = (is_congested != is_congested.shift()).cumsum()
            congested_groups = self.df[is_congested].groupby(groups[is_congested])

            # Process each continuous congestion period
            for _, group in congested_groups:
                start_time = group.index[0]
                end_time = group.index[-1]
                
                self._add_event(events, sensor_id, start_time, end_time)    

        self.events_df = pd.DataFrame(events)
        print(f"\nDetected {len(self.events_df)} total raw congestion events.")
        return self.events_df


    def analyze_events(self):
        """Provides summary statistics and insights about detected events."""
        if self.events_df.empty:
            print("\nNo events to analyze.")
            return

        print(f"\nTotal events analyzed: {len(self.events_df)}")

        print("\n--- Events by Severity ---")
        severity_summary = self.events_df.groupby('severity').agg({
            'severity': 'count',
            'duration_minutes': 'mean',
            'avg_speed': 'mean'
        }).rename(columns={'severity': 'count'}).sort_values('count', ascending=False)
        print(severity_summary)

        print("\n--- Events by Time Period ---")
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
        
        return self.events_df   # Events dataframe returned 


if __name__ == "__main__":

    dataset = MetrLA(impute_zeros=True)
    df = dataset.target
    print(df.head())

    detector = CongestionDetector(df, min_duration=3)
    
    print(f"Dataset info: {len(detector.df.columns)} sensors, {len(detector.df)} time steps")
    print(f"Date range: {detector.df.index[0]} to {detector.df.index[-1]}")
    
    events_df = detector.run_pipeline()
    
    if not events_df.empty:
        events_df.to_csv("basic_congestion_events.csv", index=False)
        print("Saved events to basic_congestion_events.csv")