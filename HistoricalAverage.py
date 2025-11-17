from tsl.datasets import MetrLA
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple
from datetime import datetime


class TrafficPredictor:
    """Traffic Predcition based on historical day-of-week and time-of-day averages."""
    
    SAMPLE_RATE = 5
    SAMPLES_PER_HOUR = 60 / SAMPLE_RATE
    SAMPLES_PER_DAY = 24 * SAMPLES_PER_HOUR
    
    def __init__(self, df: pd.DataFrame):

        self.df = self._preprocess_dataframe(df)
        self.days_averages = None


    @staticmethod
    def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        
        # Handle tuple column names for sensor_id (sensor_id, 0)
        if isinstance(df.columns[0], tuple):
            df.columns = [col[0] for col in df.columns]
            
        return df
    

    @staticmethod
    def _get_time_slot(timestamp: datetime) -> int:
        """Convert timestamp to 5-minute time slot index (0-287)"""

        return timestamp.hour * TrafficPredictor.SAMPLES_PER_HOUR + (timestamp.minute // TrafficPredictor.SAMPLE_RATE)
    

    @staticmethod
    def time_slot_to_string(time_slot: int) -> str:
        """Convert time slot index to human-readable time string"""

        hour = int(time_slot // TrafficPredictor.SAMPLES_PER_HOUR)
        minute = int((time_slot % TrafficPredictor.SAMPLES_PER_HOUR) * TrafficPredictor.SAMPLE_RATE)
        return f"{hour:02d}:{minute:02d}"


    def _accumulate_speeds(self) -> Tuple[Dict, Dict]:
        """Add observed speedss and counts for each (day, time_slot, sensor) combination"""

        speed_totals = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        day_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for sensor_id in self.df.columns:
            for timestamp, speed in zip(self.df.index, self.df[sensor_id]):
                day_name = timestamp.strftime("%A")
                time_slot = self._get_time_slot(timestamp)
                
                speed_totals[day_name][time_slot][sensor_id] += float(speed)
                day_count[day_name][time_slot][sensor_id] += 1
                
        return speed_totals, day_count
    

    def _compute_averages(self, sums: Dict, counts: Dict) -> Dict:
        """Compute average speeds from accumulated sums and counts."""

        averages = defaultdict(lambda: defaultdict(dict))
        
        for day_name in sums:
            for time_slot in sums[day_name]:
                for sensor_id in sums[day_name][time_slot]:
                    
                    total_speed = sums[day_name][time_slot][sensor_id]
                    count = counts[day_name][time_slot][sensor_id]
                    
                    avg_speed = total_speed / count if count > 0 else 0.0
                    averages[day_name][time_slot][sensor_id] = avg_speed
                    
        return averages
    

    def train(self) -> None:
        """Train the predictor by computing historical averages."""

        sums, counts = self._accumulate_speeds()
        self.days_averages = self._compute_averages(sums, counts)
        print("Computed Averages")
        

    def predict(self, day: str, time_slot: int, sensor_id: str) -> float:
        """ Predict traffic speed for a given day, time, and sensor"""

        # Could probably take hour/min as arg and convert to time slot 
        if self.days_averages is None:
            raise ValueError("Predictor must be trained before making predictions. Call train() first.")
        
        prediction = self.days_averages[day][time_slot].get(sensor_id, 0.0)
        return prediction
    

def main():

    # Load dataset
    dataset = MetrLA(impute_zeros=True)
    df: pd.DataFrame = dataset.target

    print(f"Dataset loaded: {len(df)} records, {len(df.columns)} sensors")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    predictor = TrafficPredictor(df)
    predictor.train()

    # example prediction
    day = "Monday"
    time_slot = 144  # 12:00 PM
    sensor_id = "767509"
    
    prediction = predictor.predict(day, time_slot, sensor_id)
    time_str = TrafficPredictor.time_slot_to_string(time_slot)
    
    print(f"Prediction Results")
    print(f"Day:        {day}")
    print(f"Time:       {time_str} (slot {time_slot})")
    print(f"Sensor:     {sensor_id}")
    print(f"Predicted:  {prediction:.2f} mph")


if __name__ == "__main__":
    main()