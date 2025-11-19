import os
import gc
import datetime
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsl.datasets import MetrLA

# ==========================================
#               CONFIGURATION
# ==========================================
# Change parameters here to adjust processing and visualization
CONFIG = {
    "IO": {
        "BASE_OUTPUT_DIR": "sensor_minimaps",
        "TARGET_CATEGORY": "Monday",  # Options: "Monday", "Tuesday", "Weekend", "All", etc.
    },
    "GRID": {
        "ROWS": 5,
        "COLS": 3,
        "FIG_SIZE": (15, 20),
    },
    "ANALYSIS": {
        "CONGESTION_PCT": 0.50,       # Flag if speed < 50% of free-flow
        "MIN_EVENT_DURATION": 15,     # Minutes
        "IMPUTE_ZEROS": True,
    },
    "VISUALIZATION": {
        "SHOW_MEAN": True,
        "SHOW_MEDIAN": True,
        "SHOW_ACTUALS": True,
        "SHOW_EVENTS": True,
        "SHOW_UNCERTAINTY": False,    
        "COLORS": {
            "MEAN": "red",
            "MEDIAN": "orange",
            "ACTUALS": "lightgray",
            "EVENT_FILL": "red",
            "UNCERTAINTY_STD": "red",
            "UNCERTAINTY_RANGE": "gray",
        }
    }
}

DAY_CATEGORIES = {
    "Monday":    [0],
    "Tuesday":   [1],
    "Wednesday": [2],
    "Thursday":  [3],
    "Friday":    [4],
    "Weekend":   [5, 6]
}

# ==========================================
#             DATA PROCESSING
# ==========================================

def time_to_minutes(t: datetime.time) -> int:
    """Converts a datetime.time object to minutes since midnight."""
    return t.hour * 60 + t.minute

def load_dataset() -> Tuple[pd.DataFrame, pd.Index]:
    """Loads the full MetrLA dataset."""
    print("Loading full MetrLA dataset...")
    metr_la = MetrLA(impute_zeros=CONFIG["ANALYSIS"]["IMPUTE_ZEROS"])
    full_df = metr_la.target
    full_df.index = pd.to_datetime(full_df.index)
    
    sensor_ids = full_df.columns
    print(f"Dataset loaded. Found {len(sensor_ids)} sensors.")
    return full_df, sensor_ids

def get_sensor_data(source_df: pd.DataFrame, sensor_id: str) -> pd.DataFrame:
    """Slices data for a specific sensor."""
    return pd.DataFrame({
        'speed': source_df[sensor_id].values,
        'time_of_day': source_df.index.time
    })

def calculate_daily_pattern(actual_values: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregates speed data to find mean, median, std, min, and max patterns."""
    # Aggregation
    pattern_df = actual_values.groupby('time_of_day')['speed'].agg(
        ['mean', 'median', 'std', 'min', 'max']
    ).reset_index()

    # Add numerical minute column for plotting
    pattern_df['time_minutes'] = pattern_df['time_of_day'].apply(time_to_minutes)
    actual_values['time_minutes'] = actual_values['time_of_day'].apply(time_to_minutes)
    
    return actual_values, pattern_df

# ==========================================
#             PLOTTING LOGIC
# ==========================================

def highlight_congestion_events(ax: plt.Axes, pattern_df: pd.DataFrame):
    """
    Algorithmic Labeling: Highlights periods where median speed drops below
    a dynamic percentage of the sensor's free-flow speed.
    """
    # 1. Define Dynamic Threshold
    free_flow_speed = pattern_df['median'].max()
    threshold = free_flow_speed * CONFIG["ANALYSIS"]["CONGESTION_PCT"]
    
    # 2. Identify Congestion Blocks
    is_congested = pattern_df['median'] < threshold
    groups = is_congested.ne(is_congested.shift()).cumsum()
    congested_groups = pattern_df[is_congested].groupby(groups)

    # 3. Annotate Events
    for _, group in congested_groups:
        if group.empty: continue
            
        start = group['time_minutes'].iloc[0]
        end = group['time_minutes'].iloc[-1]
        duration = end - start
        
        if duration >= CONFIG["ANALYSIS"]["MIN_EVENT_DURATION"]:
            # Find the point of maximum congestion (lowest speed)
            min_idx = group['median'].idxmin()
            min_row = group.loc[min_idx]
            
            # Draw Red Band
            ax.axvspan(start, end, color=CONFIG["VISUALIZATION"]["COLORS"]["EVENT_FILL"], alpha=0.1)
            
            # Draw Text Annotation
            ax.annotate(
                f"{int(min_row['median'])} mph", 
                xy=(min_row['time_minutes'], min_row['median']), 
                xytext=(min_row['time_minutes'], min_row['median'] - 12),
                arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.7),
                ha='center', fontsize=7, color='darkred'
            )

def draw_sensor_subplot(ax: plt.Axes, actual: pd.DataFrame, pattern: pd.DataFrame, sensor_id: str):
    """Draws the speed pattern, uncertainty bands, and events on a single subplot."""
    vals = CONFIG["VISUALIZATION"]
    colors = vals["COLORS"]
    x_vals = pattern['time_minutes'].values  

    if vals["SHOW_UNCERTAINTY"]:
        # Min/Max Range
        ax.fill_between(x_vals, pattern['min'].values, pattern['max'].values,
                        color=colors["UNCERTAINTY_RANGE"], alpha=0.1, label='Min-Max')
        # Std Dev Range
        upper = pattern['mean'] + pattern['std']
        lower = pattern['mean'] - pattern['std']
        ax.fill_between(x_vals, lower.values, upper.values,
                        color=colors["UNCERTAINTY_STD"], alpha=0.15, label='Std Dev')

    if vals["SHOW_ACTUALS"]:
        
        ax.plot(actual['time_minutes'].values, actual['speed'].values,
                alpha=0.8, linewidth=0.8, 
                color=colors["ACTUALS"], zorder=1)

    if vals["SHOW_MEAN"]:
        ax.plot(x_vals, pattern['mean'].values, 
                color=colors["MEAN"], 
                linewidth=1, label='Mean', zorder=2)
    
    if vals["SHOW_MEDIAN"]:
        ax.plot(x_vals, pattern['median'].values, 
                color=colors["MEDIAN"], 
                linewidth=1, label='Median', zorder=3)

    if vals["SHOW_EVENTS"]:
        highlight_congestion_events(ax, pattern)

    ax.set_title(sensor_id, fontsize=10)
    ax.grid(alpha=0.3)
    
    # X-Axis
    ax.set_xticks([0, 360, 720, 1080, 1440])
    ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'], fontsize=8)
    
    # Y-Axis
    ax.set_ylim(0, 80)
    ax.tick_params(axis='y', labelsize=8)
    
    
#=========================================
#           MAIN PROCESSING LOOP
#=========================================

def generate_figure_batch(category_dir: str, batch_ids: list, full_df: pd.DataFrame, 
                          start_idx: int, end_idx: int, category_name: str):
    """Generates and saves a single PNG file containing a grid of sensor plots."""
    
    fname = f"sensors_{start_idx:03d}_to_{end_idx-1:03d}.png"
    save_path = os.path.join(category_dir, fname)
    print(f"  Generating {fname}...")

    rows, cols = CONFIG["GRID"]["ROWS"], CONFIG["GRID"]["COLS"]
    fig, axes = plt.subplots(rows, cols, figsize=CONFIG["GRID"]["FIG_SIZE"], constrained_layout=True)
    axes_flat = axes.flatten()

    for i, sensor_id in enumerate(batch_ids):
        ax = axes_flat[i]
        # 1. Slice & Calculate
        sensor_vals = get_sensor_data(full_df, sensor_id)
        actual_data, pattern_df = calculate_daily_pattern(sensor_vals)
        
        # 2. Draw
        draw_sensor_subplot(ax, actual_data, pattern_df, sensor_id)

    # Hide unused subplots
    for i in range(len(batch_ids), rows * cols):
        axes_flat[i].axis('off')

    fig.suptitle(f"{category_name} Patterns: {start_idx}-{end_idx-1}", fontsize=16, y=1.02)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    # Memory Cleanup
    plt.close(fig)
    plt.clf()
    gc.collect()

def process_category(category_name: str, day_indices: list, full_df: pd.DataFrame, sensor_ids: pd.Index):
    """Filters data for a specific category (e.g., 'Monday') and runs the plotting batches."""
    print(f"\nProcessing Category: {category_name}...")

    # 1. Filter Data by Day
    mask = full_df.index.dayofweek.isin(day_indices)
    day_specific_df = full_df[mask]

    # 2. Prepare Output Directory
    category_dir = os.path.join(CONFIG["IO"]["BASE_OUTPUT_DIR"], category_name)
    os.makedirs(category_dir, exist_ok=True)

    # 3. Loop in Batches
    plots_per_fig = CONFIG["GRID"]["ROWS"] * CONFIG["GRID"]["COLS"]
    total_sensors = len(sensor_ids)

    for start in range(0, total_sensors, plots_per_fig):
        end = min(start + plots_per_fig, total_sensors)
        batch_ids = sensor_ids[start:end]
        
        generate_figure_batch(category_dir, batch_ids, day_specific_df, start, end, category_name)

def main():
    full_df, sensor_ids = load_dataset()
    target = CONFIG["IO"]["TARGET_CATEGORY"]

    # Determine which categories to run
    if target == "All":
        categories_to_run = DAY_CATEGORIES
    elif target in DAY_CATEGORIES:
        categories_to_run = {target: DAY_CATEGORIES[target]}
    else:
        print(f"Error: '{target}' is not a valid category.")
        return

    # Run Pipeline
    for name, indices in categories_to_run.items():
        process_category(name, indices, full_df, sensor_ids)

    print(f"\nFinished processing: {target}")

if __name__ == "__main__":
    main()