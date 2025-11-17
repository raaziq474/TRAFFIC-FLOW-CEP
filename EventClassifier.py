from typing import Dict, Optional
import pandas as pd

class EventClassifier:
    """Handles the classification logic for event types"""

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
    def _classify_severity(speed: float) -> Optional[str]:

        """Classifies congestion severity based on average speed."""
        if speed < 30:
            return "heavy"
        elif speed < 40:
            return "moderate"
        elif speed < 50:
            return "light"
        else:
            return None # No congestion

    @staticmethod
    def _classify_duration(duration_intervals: int) -> str:
        """Classifies event duration based on the number of 5-minute intervals."""

        if duration_intervals < 6:
            return "short"
        elif duration_intervals <= 12:
            return "medium"
        else:
            return "long"


    @staticmethod
    def classify_event_types(speed: float, duration_intervals: int, start_time: pd.Timestamp) -> Optional[Dict[str, str]]:
        """
        Classifies the event into severity, duration_type, and time_period.
        """
        severity = EventClassifier._classify_severity(speed)
        
        if severity is None:
            return None # No congestion - might need to change 
        
        duration_type = EventClassifier._classify_duration(duration_intervals)
        time_period = EventClassifier._classify_time_period(start_time)
        
        return {
            "severity": severity,
            "duration_type": duration_type,
            "time_period": time_period
        }