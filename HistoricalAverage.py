from tsl.datasets import MetrLA
import pandas as pd
from collections import defaultdict 
from datetime import datetime
import numpy as np
from tqdm import tqdm


class HistoricalAverageTraffic:
    """Traffic Predcition based on historical day-of-week and time-of-day averages."""
    
    SAMPLE_RATE = 5
    SAMPLES_PER_HOUR = 60 // SAMPLE_RATE
    SAMPLES_PER_DAY = 24 * SAMPLES_PER_HOUR
    
    def __init__(self, df: pd.DataFrame):

        self.df = self._preprocess_dataframe(df)
        self.speed_statistics = None


    @staticmethod
    def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        
        # Handle tuple column names for sensor_id (sensor_id, 0)
        if isinstance(df.columns[0], tuple):
            df.columns = [col[0] for col in df.columns]
            
        return df
    

    @staticmethod
    def _get_time_slot(timestamp: datetime) -> int:
        """Convert timestamp to 5-minute time slot index (0-287)"""

        time_slot = timestamp.hour * HistoricalAverageTraffic.SAMPLES_PER_HOUR + (timestamp.minute // HistoricalAverageTraffic.SAMPLE_RATE)
        return time_slot
    

    @staticmethod
    def time_slot_to_string(time_slot: int) -> str:
        """Convert time slot index to human-readable time string"""

        hour = int(time_slot // HistoricalAverageTraffic.SAMPLES_PER_HOUR)
        minute = int((time_slot % HistoricalAverageTraffic.SAMPLES_PER_HOUR) * HistoricalAverageTraffic.SAMPLE_RATE)
        return f"{hour:02d}:{minute:02d}"


    def _accumulate_speeds(self) -> dict:
        """Store lists of speeds for each (day, time_slot, sensor) combination"""
        speeds = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        total_iterations = len(self.df.columns) * len(self.df.index)    # sensors * timesteps 
        
        # Update less frequently using mininterval
        pbar = tqdm(total=total_iterations, desc="Accumulating speeds", mininterval=0.5)
        
        for sensor_id in self.df.columns:
            for timestamp, speed in zip(self.df.index, self.df[sensor_id]):

                day_name = timestamp.strftime("%A")
                time_slot = self._get_time_slot(timestamp)
                speeds[day_name][time_slot][sensor_id].append(float(speed))
                pbar.update(1)
        
        pbar.close()
        return speeds


    def _compute_statistics(self, speeds) -> dict:
        """Compute mean, stdev, min, max, lower, upper."""
        
        results = defaultdict(lambda: defaultdict(dict))
        
        # Days * Time slots * Sensors 
        total_iterations = len(speeds) * self.SAMPLES_PER_DAY * len(self.df.columns)    

        pbar = tqdm(total=total_iterations, desc="Computing Sensor Statistics")
        
        for day in speeds:
            for time_slot in speeds[day]:
                for sensor, values in speeds[day][time_slot].items():
                    
                    arr = np.array(values)
                    results[day][time_slot][sensor] = {     # could add custom stats 
                        "mean": arr.mean(),
                        "median": np.median(arr),
                        "stdev": arr.std(),
                        "min": arr.min(),
                        "max": arr.max()
                    }
                    pbar.update(1)
        
        pbar.close()
        return results
    

    def train(self) -> None:
        """'Train' the predictor by segmenting sensor values between day and time and computing a distribution on them"""

        print("Computing Historical Averages...")
        speeds = self._accumulate_speeds()
        self.speed_statistics = self._compute_statistics(speeds)
        print("Finished Training")
        

    def predict_distribution(self, day: str | datetime, time_slot: str | int | datetime, sensor_id: str) -> dict:
        """Predict traffic speed for a given day, time, and sensor"""

        if self.speed_statistics is None:
            raise ValueError("Predictor must be trained before making predictions. Call train() first")

        if isinstance(day, datetime):
            day = day.strftime("%A")                     # get day name if day is timestamp 

        if isinstance(time_slot, str):
            dtime = pd.to_datetime(time_slot)            # convert string to datetime
            time_slot = self._get_time_slot(dtime)       # use date time to get time slot 

        elif isinstance(time_slot, datetime):
            time_slot = self._get_time_slot(time_slot)

        prediction = self.speed_statistics[day][time_slot][sensor_id]
        return prediction
    

    def predict_day_distribution(self, day: str, sensor_id: str) -> dict:
        """Predict aggregated traffic statistics for a given day and sensoracross the entire day (all 288 time slots)."""

        if self.speed_statistics is None:
            raise ValueError("Predictor must be trained before making predictions. Call train() first")

        if isinstance(day, datetime):
            day = day.strftime("%A")                    # get day name if day is timestamp 

        # these statistics must already be defined in the _compute_statistices()
        statistics = {
            "mean": [],
            "median": [],
            "stdev": [],
            "min": [],
            "max": []
        }

        # Collect all distributions across the day
        for time_slot in self.speed_statistics[day]:
            if sensor_id in self.speed_statistics[day][time_slot]:
                stats = self.speed_statistics[day][time_slot][sensor_id]

                for key in statistics:
                    statistics[key].append(stats[key])

        # Aggregate the statistics across the full day
        return {
            "mean": float(np.mean(statistics["mean"])),
            "median": float(np.mean(statistics["median"])),
            "stdev": float(np.mean(statistics["stdev"])),
            "min": float(np.min(statistics["min"])),
            "max": float(np.max(statistics["max"]))
        }

    

def main():

    # example usage using historical average traffic preditor
    dataset = MetrLA(impute_zeros=True)
    df: pd.DataFrame = dataset.target
    
    print(f"Dataset loaded: {len(df)} records, {len(df.columns)} sensors")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    predictor = HistoricalAverageTraffic(df)
    predictor.train()

    # example prediction
    day = "Thursday"
    time = "12:30" # can provide either hour/minute or time slot 
    sensor_id = "767509"
    
    prediction = predictor.predict_day_distribution(day, sensor_id)
    
    print(f"\nPrediction Results")
    print(f"Day: {day}")
    print(f"Time: {time}")
    print(f"Sensor: {sensor_id}")
    
    for key, value in prediction.items():
        print(f"{key}: {value}")




if __name__ == "__main__":
    main()