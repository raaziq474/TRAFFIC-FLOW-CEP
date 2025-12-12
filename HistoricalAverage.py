import pandas as pd
from collections import defaultdict 
from datetime import datetime
import numpy as np
from tqdm import tqdm
import data_utils


#################
# Historical Average Model for traffic regression
#################

class HistoricalAverageTraffic:
    """Traffic Predcition based on historical day-of-week and time-of-day averages."""
    
    SAMPLE_RATE = 5
    SAMPLES_PER_HOUR = 60 // SAMPLE_RATE
    SAMPLES_PER_DAY = 24 * SAMPLES_PER_HOUR
    
    def __init__(self, df: pd.DataFrame):

        self.df = df
        self.speed_statistics = None


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
        
        print("\nTraining Historical Average Model...")
        pbar = tqdm(total=total_iterations, desc="Segmenting Sensors", mininterval=0.5)
        
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

        pbar = tqdm(total=total_iterations, desc="Computing Sensor Stats")
        
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
    

    def _compute_daily_statistics(self) -> dict:
        """Precompute per-day aggregated stats for each sensor"""

        daily_stats = defaultdict(lambda: defaultdict(dict))

        total_iterations = len(self.speed_statistics) * len(self.df.columns)
        pbar = tqdm(total=total_iterations, desc="Computing Daily Sensor Stats", mininterval=0.5)

        for day in self.speed_statistics:
            for sensor_id in self.df.columns:
                stats = {
                    "mean": [],
                    "median": [],
                    "stdev": [],
                    "min": [],
                    "max": []
                }

                for time_slot in self.speed_statistics[day]:
                    if sensor_id in self.speed_statistics[day][time_slot]:
                        s = self.speed_statistics[day][time_slot][sensor_id]
                        for key in stats:
                            stats[key].append(s[key])

                # Store only if data exists
                if stats["mean"]:
                    daily_stats[day][sensor_id] = {
                        "mean": float(np.mean(stats["mean"])),
                        "median": float(np.mean(stats["median"])),
                        "stdev": float(np.mean(stats["stdev"])),
                        "min": float(np.min(stats["min"])),
                        "max": float(np.max(stats["max"]))
                    }

                pbar.update(1)

        pbar.close()
        return daily_stats


    def train(self) -> None:
        """'Train' the predictor by segmenting sensor values between day and time and computing a distribution on them"""

        speeds = self._accumulate_speeds()
        self.speed_statistics = self._compute_statistics(speeds)
        self.daily_statistics = self._compute_daily_statistics()
        

    def predict_timestep_distribution(self, day: str | datetime, time_slot: str | int | datetime, sensor_id: str) -> dict:
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
            day = day.strftime("%A")

        return self.daily_statistics[day][sensor_id]

    
def main():

    # example usage using historical average traffic predictor
    df: pd.DataFrame = data_utils.load_csv("la")
    
    print(f"Dataset loaded: {len(df)} records, {len(df.columns)} sensors")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    predictor = HistoricalAverageTraffic(df)
    predictor.train()

    # example prediction
    day = "Thursday"
    time = "12:30"          # can provide either hour/minute or time slot 
    sensor_id = "767509"
    
    prediction = predictor.predict_timestep_distribution(day, time, sensor_id)
    
    print(f"\nPrediction Results")
    print(f"Day: {day}")
    print(f"Time: {time}")
    print(f"Sensor: {sensor_id}")
    
    for key, value in prediction.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()