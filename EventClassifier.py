from typing import Dict, Optional
import pandas as pd


#################
# Event Classifier Module - contains several classification methods based on event properties
#################


# NOTE: Dont really need a class here 
class EventClassifier:
    """Handles the classification logic for event types"""
    
    @staticmethod
    def _classify_severity(speed: float, day_stats: dict) -> Optional[str]:
        """Classifies congestion severity based on average speed,
           uses min max normalization and threshold based system"""
        
        day_min = day_stats["min"]
        day_max = day_stats["max"]
        
        if day_max == day_min:     # Avoid division by zero if sensor down entire day
            return None

        # Normalize speed to 0 (min) - 1 (max)
        norm_speed = (speed - day_min) / (day_max - day_min)

        # Classify based on normalized percentage, name is "tertile" not quartile lol 
        if norm_speed < 0.33:
            return "heavy"
        elif norm_speed < 0.66:
            return "moderate"
        else:
            return "light"  # 0.6666 - 1


    @staticmethod
    def _classify_time_period(timestamp: pd.Timestamp) -> str:
        """Classifies the event/timestamp into hour of day it occurs"""

        hour = timestamp.hour
        
        if 6 <= hour < 10:
            return "morning_rush_hour"
        elif 16 <= hour < 20:
            return "evening_rush_hour"
        else:
            return "off_peak"


    @staticmethod
    def _classify_duration(duration_intervals: int) -> str:
        """Classifies event duration based on the number of 5-minute intervals"""
    
        if duration_intervals < 6:         # 30 minutes
            return "short"
        
        elif duration_intervals <= 12:     # 60 minutes
            return "medium"
        else:
            return "long"                  # more than 1 hour
    

    @staticmethod
    def _classify_day_of_week(date_time: pd.Timestamp) -> str:
        """Classifies event based on the day of week"""

        return date_time.strftime("%A") # retuns string fromat of day eg. 0 -> Monday 


    @staticmethod
    def get_event_labels(speed: float, duration_intervals: int, start_time: pd.Timestamp, day_stats: dict) -> Optional[Dict[str, str]]:
        """Classifies the event based on multiple fields, and returns classification types"""

        severity = EventClassifier._classify_severity(speed, day_stats)
        
        if severity is None:
            return None
        
        duration_type = EventClassifier._classify_duration(duration_intervals)
        day_of_week = EventClassifier._classify_day_of_week(start_time)
        
        return {
            "severity": severity,
            "duration_type": duration_type,
            "day_of_week": day_of_week
        }
    

    @staticmethod
    def _classify_congestion_type(speed, timestamp, sensor_id, predictor):
        """Returns 'random', 'peak', or None."""

        day_stats = predictor.predict_day_distribution(timestamp, sensor_id)
        timeperiod_stats = predictor.predict_timestep_distribution(timestamp, timestamp, sensor_id)

        random_threshold = timeperiod_stats["mean"] - timeperiod_stats["stdev"]

        if speed < random_threshold:
            return "random"
        
        elif speed < day_stats["mean"]:
            return "peak"
        
        return None


    @staticmethod
    def build_event(series: pd.Series, sensor_id: str, start_time, end_time, event_type, predictor):
        """Builds the final event dict of labels and classification types for a complete event"""

        event_series = series.loc[start_time:end_time]
        duration_intervals = len(event_series)
        avg_speed = round(float(event_series.mean()), 2)
        day_stats = predictor.predict_day_distribution(start_time, sensor_id)

        classification = EventClassifier.get_event_labels(avg_speed, duration_intervals, start_time, day_stats)

        if classification is None:
            return None

        event_dict = {
            "sensor_id": sensor_id,
            "start_time": start_time,
            "end_time": end_time,
            "avg_speed": avg_speed,
            "duration_minutes": duration_intervals * 5,
            "type_detected": event_type,
        }
        
        event_dict.update(classification)
        return event_dict
