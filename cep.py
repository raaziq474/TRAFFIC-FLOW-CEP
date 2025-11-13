import pandas as pd
import numpy as np
import networkx as nx
from tsl.datasets import MetrLA
from datetime import time


def classify_time_period(timestamp):
    # TODO: combine with classify event type 
    hour = timestamp.hour
    
    if 6 <= hour < 10:
        return "morning_peak"
    elif 16 <= hour < 20:
        return "evening_peak"
    else:
        return "off_peak"


def classify_event_type(speed, duration_intervals, time_period):

    # Severity classification
    if speed < 30:
        severity = "heavy"
    elif speed < 40:
        severity = "moderate"
    elif speed < 50:
        severity = "light"
    else:
        return None  # no congestion 
    
    # Duration classification
    if duration_intervals < 6:
        duration = "short"
    elif duration_intervals <= 12:
        duration = "medium"
    else:
        duration = "long"
    
    # Combine into event type
    event_type = f"{severity}_{time_period}_{duration}"
    
    return event_type


def add_event(events, df, sensor_id, start_time, end_time, min_duration=3):
    """Record a congestion event if it meets minimum duration and classification rules."""

    duration_intervals = len(df.loc[start_time:end_time]) - 1

    if duration_intervals < min_duration:
        return

    speeds = df.loc[start_time:end_time, sensor_id]
    avg_speed = speeds.mean()                                                       # average speed across event 
    time_period = classify_time_period(start_time)                                  # duration of event 
    event_type = classify_event_type(avg_speed, duration_intervals, time_period)    # classify event based on speed, duration and time 

    if event_type:
        events.append({
            "sensor_id": sensor_id,
            "start_time": start_time,
            "end_time": end_time,
            "event_type": event_type,
            "avg_speed": avg_speed,
            "duration_minutes": duration_intervals * 5,
            "time_period": time_period
        })


def detect_multi_type_events(df, min_duration=3):
    """
    Detect congestion events based on severity, duration, and time period.
    Each event must last at least min_duration * 5 minutes.
    """
    events = []

    for sensor_id in df.columns:
        is_congested = df[sensor_id] < 50 # congestion only considered below 50

        # Identify consecutive runs of congestion
        groups = (is_congested != is_congested.shift()).cumsum()
        congested_groups = df[is_congested].groupby(groups[is_congested])

        for _, group in congested_groups:
            if len(group) >= min_duration:
                start_time = group.index[0]
                end_time = group.index[-1]
                add_event(events, df, sensor_id, start_time, end_time, min_duration)

    return pd.DataFrame(events)


def run_pipeline(df, id_to_node, min_duration=3):
    
    events = detect_multi_type_events(df, min_duration)
    print(f"\nDetected {len(events)} total congestion events.")
    
    # Print event type distribution
    if len(events) > 0:
        print("\nEvent type distribution:")
        print(events['event_type'].value_counts())
        print("\nTime period distribution:")
        print(events['time_period'].value_counts())

    enriched_events = []

    for i, e in events.iterrows():
        sensor_id = e["sensor_id"]
        sensor_node = id_to_node.get(sensor_id)

        if sensor_node is None:
            continue

        enriched_events.append({
            "sensor_id": sensor_id,
            "start_time": e["start_time"],
            "end_time": e["end_time"],
            "event_type": e["event_type"],
            "avg_speed": e["avg_speed"],
            "duration_minutes": e["duration_minutes"],
            "time_period": e["time_period"]
        })

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} events...")

    events_df = pd.DataFrame(enriched_events)
    print("\nProcessing complete.")
    return events_df


def analyze_events(events_df):
    """Provide summary statistics and insights about detected events."""
    
    print(f"\nTotal events detected: {len(events_df)}")
    print(f"Average event duration: {events_df['duration_minutes'].mean():.1f} minutes")
    print(f"Average speed during events: {events_df['avg_speed'].mean():.1f} mph")
    
    # Group by event type
    print("\n--- Events by Type ---")
    type_summary = events_df.groupby('event_type').agg({
        'event_type': 'count',
        'duration_minutes': 'mean',
        'avg_speed': 'mean'
    }).rename(columns={'event_type': 'count'})
    type_summary = type_summary.sort_values('count', ascending=False)
    print(type_summary)
    
    # Group by time period
    print("\n--- Events by Time Period ---")
    period_summary = events_df.groupby('time_period').agg({
        'time_period': 'count',
        'duration_minutes': 'mean',
        'avg_speed': 'mean'
    }).rename(columns={'time_period': 'count'})
    print(period_summary)

    return 

if __name__ == "__main__":
    
    dataset = MetrLA(impute_zeros=True)     # Load METR-LA dataset from TSL
    df = dataset.target                     # traffic DataFrame

    # TODO: Select / Change partition here ?

    sensor_ids = df.columns.tolist()
    id_to_node = {sid: i for i, sid in enumerate(sensor_ids)}

    print(f"Dataset info: {len(sensor_ids)} sensors, {len(df)} time steps")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Run CEP pipeline with multi-type event detection
    events_df = run_pipeline(df, id_to_node, min_duration=3)
    
    analyze_events(events_df)

    events_df.to_csv("multi_type_congestion_events.csv", index=False)
    print("Saved events to multi_type_congestion_events.csv")
