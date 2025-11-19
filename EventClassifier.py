from typing import Dict, Optional
import pandas as pd


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

        if norm_speed < 0.33:
            return "heavy"
        elif norm_speed < 0.66:
            return "moderate"
        else:           # norm_speed < 1.0
            return "light"


    @staticmethod
    def _classify_time_period(timestamp: pd.Timestamp) -> str:
        """Classifies the timestamp into morning_peak, evening_peak, or off_peak."""

        hour = timestamp.hour
        
        if 6 <= hour < 10:
            return "morning_peak"
        elif 16 <= hour < 20:
            return "evening_peak"
        else:
            return "off_peak"


    @staticmethod
    def _classify_duration(duration_intervals: int) -> str:
        """Classifies event duration based on the number of 5-minute intervals."""
    
        # TODO: Need average duration length of event 
        if duration_intervals < 6:
            return "short"
        elif duration_intervals <= 12:
            return "medium"
        else:
            return "long"
    

    @staticmethod
    def _classify_day_of_week(date_time: pd.Timestamp) -> str:
        """Classifies event based on the day of week"""

        return date_time.strftime("%A") # retuns string fromat of day eg. Monday 


    @staticmethod
    def classify_event_types(speed: float, duration_intervals: int, start_time: pd.Timestamp, day_stats: dict) -> Optional[Dict[str, str]]:
        """
        Classifies the event into severity, duration_type, and time_period.
        """

        severity = EventClassifier._classify_severity(speed, day_stats)
        
        if severity is None:
            return None # No congestion - might need to change 
        
        duration_type = EventClassifier._classify_duration(duration_intervals)
        time_period = EventClassifier._classify_time_period(start_time)
        day_of_week = EventClassifier._classify_day_of_week(start_time)
        
        return {
            "severity": severity,
            "duration_type": duration_type,
            "time_period": time_period,
            "day_of_week": day_of_week
        }