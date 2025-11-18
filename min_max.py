import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from tsl.datasets import MetrLA

# Use gc if needed for memory management- uncomment the import below 
# And the gc.collect() line later in the main() function
# import gc

# Minimap grid configuration
GRID_ROWS = 5
GRID_COLS = 3
PLOTS_PER_FIGURE = GRID_ROWS * GRID_COLS
BASE_OUTPUT_DIR = "sensor_minimaps"

# Define Day categories: Name -> List of Day Integers
DAY_CATEGORIES = {
    "Monday":    [0],
    "Tuesday":   [1],
    "Wednesday": [2],
    "Thursday":  [3],
    "Friday":    [4],
    "Weekend":   [5, 6] 
}

# Constants to control what to show on the minimaps
SHOW_MEAN = True
SHOW_MEDIAN = True
SHOW_ACTUALS = True

def time_to_minutes(t: datetime.time) -> int:
    """Converts a datetime.time object to minutes since midnight."""
    return t.hour * 60 + t.minute

def load_full_dataset():
    """
    Loads the full METR-LA dataset and returns the DataFrame and sensor IDs.
    """
    metr_la = MetrLA(impute_zeros=True)
    full_df = metr_la.target
    full_df.index = pd.to_datetime(full_df.index)
    all_sensor_ids = full_df.columns
    print(f"Dataset loaded. Found {len(all_sensor_ids)} sensors.")
    return full_df, all_sensor_ids

def get_sensor_data(source_df, sensor_id):
    """
    Slices data for a specific sensor from the provided DataFrame and returns a DataFrame
    with 'speed' and 'time_of_day' columns.
    """
    actual_values = pd.DataFrame({
        'speed': source_df[sensor_id].values,
        'time_of_day': source_df.index.time
    })
    return actual_values

def calculate_daily_pattern(actual_values):
    """Calculates the daily pattern (mean and median) for the given sensor data."""
    pattern_df = actual_values.groupby('time_of_day')['speed'].agg(['mean', 'median']).reset_index()
    pattern_df['time_minutes'] = pattern_df['time_of_day'].apply(time_to_minutes)
    actual_values['time_minutes'] = actual_values['time_of_day'].apply(time_to_minutes)
    return actual_values, pattern_df

def draw_pattern_on_ax(ax, actual_data, pattern_data, sensor_id):
    """Draw the plot on a specific subplot axis."""
    if SHOW_ACTUALS:
        ax.plot(
            actual_data['time_minutes'].values,
            actual_data['speed'].values,
            alpha=0.8, linewidth=0.8, color='lightgray'
        )

    if SHOW_MEAN:
        ax.plot(
            pattern_data['time_minutes'].values,
            pattern_data['mean'].values,
            color='red', linewidth=1, label='Mean'
        )
    
    if SHOW_MEDIAN:
        ax.plot(
            pattern_data['time_minutes'].values,
            pattern_data['median'].values,
            color='orange', linewidth=1, label='Median'
        )

    ax.set_title(sensor_id, fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks([0, 6 * 60, 12 * 60, 18 * 60])
    ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00'], fontsize=8)
    ax.set_ylim(0, 80)
    ax.tick_params(axis='y', labelsize=8)

def main():
    # CONFIGURATION: SELECT WHICH DAY TO RUN
    # Options: "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Weekend"
    # Set this to "All" if you want to try generating for all days in one go (may take a longer time)
    
    TARGET_CATEGORY = "Weekend" 
    

    # Load Data ONCE
    full_df, all_sensor_ids = load_full_dataset()
    total_sensors = len(all_sensor_ids)

    # Logic to determine which categories to process
    if TARGET_CATEGORY == "All":
        categories_to_process = DAY_CATEGORIES
    elif TARGET_CATEGORY in DAY_CATEGORIES:
        # Create a dictionary with JUST the one selected day
        categories_to_process = {TARGET_CATEGORY: DAY_CATEGORIES[TARGET_CATEGORY]}
    else:
        print(f"Error: '{TARGET_CATEGORY}' is not a valid category name.")
        return

    # Loop through the selected categories
    for category_name, day_indices in categories_to_process.items():
        print(f"\nProcessing Category: {category_name}...")
        
        # This checks every timestamp in the index. It returns a Series of True/False values.
        # True if the day (0=Mon, 6=Sun) is in our target list; otherwise False.
        mask = full_df.index.dayofweek.isin(day_indices)
        
        # Filter the full dataset to create a subset containing only those days
        day_specific_df = full_df[mask]

        # Create the folder
        category_dir = os.path.join(BASE_OUTPUT_DIR, category_name)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)

        # Loop through Sensors
        for start_index in range(0, total_sensors, PLOTS_PER_FIGURE):
            # Determine the batch of sensors for this figure
            end_index = min(start_index + PLOTS_PER_FIGURE, total_sensors)
            sensor_batch_ids = all_sensor_ids[start_index:end_index]
            
            # Generate the file name
            file_name = f"sensors_{start_index:03d}_to_{end_index-1:03d}.png"
            save_path = os.path.join(category_dir, file_name)
            
            # Check statement for progress
            print(f"  Generating {category_name} map: {file_name}")

            fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(15, 20), constrained_layout=True)
            axes_flat = axes.flatten()

            for i, sensor_id in enumerate(sensor_batch_ids):
                ax = axes_flat[i]
                actual_values = get_sensor_data(day_specific_df, sensor_id)
                actual_data, pattern_df = calculate_daily_pattern(actual_values)
                draw_pattern_on_ax(ax, actual_data, pattern_df, sensor_id)

            for i in range(len(sensor_batch_ids), PLOTS_PER_FIGURE):
                axes_flat[i].axis('off')

            fig.suptitle(f"{category_name} Patterns: {start_index}-{end_index-1}", fontsize=16, y=1.02)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            
            plt.close(fig)    
            plt.clf()
            # Free up memory(To be able to run my laptop)         
            # gc.collect()      

    print(f"\n Finished processing: {TARGET_CATEGORY}")

if __name__ == "__main__":
    main()