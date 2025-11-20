import pandas as pd
from typing import List

class CongestionChainDetector:

    @staticmethod
    def find_causal_events(events_df: pd.DataFrame, adj_matrix, sensor_ids: List[str],
                       time_window_minutes: int = 15, min_connectivity: float = 0.1) -> pd.DataFrame:
        """
        Identifies potential cuase and effect relationship between sensors
        
        Args:
            events_df: DataFrame with congestion events (must have: sensor_id, start_time, end_time)
            sensor_ids: List of sensor IDs corresponding to adj_matrix rows/cols (adj matrix has no sensor ids)
            time_window_minutes: Maximum time delay to consider causality (default: 15 min)
            min_connectivity: Minimum weight between sensors to consider them connected (default: 0.1)
        
        Returns:
            DataFrame with causal relationships
        """
        
        if not isinstance(adj_matrix, tuple) or len(adj_matrix) != 2:
                raise ValueError("Invalid adj_matrix format provided: valid format is (edge_index, edge_weight)")
            
        edge_index, edge_weight = adj_matrix
        
        # Build connectivity lookup dictionary
        connectivity = {}
        for i in range(edge_index.shape[1]):
            source = int(edge_index[0, i])
            dest = int(edge_index[1, i])
            weight = float(edge_weight[i])
            connectivity[(source, dest)] = weight
        
        # Map sensor IDs to indices
        sensor_to_idx = {str(sid): idx for idx, sid in enumerate(sensor_ids)}
        
        # Add event IDs and sort by time
        events = events_df.copy()
        events['event_id'] = range(len(events))
        events = events.sort_values('start_time').reset_index(drop=True)
        
        causal_relationships = []
        
        # For each event, find potential downstream effects
        for _, cause_event in events.iterrows():
            cause_sensor = str(cause_event['sensor_id'])
            
            if cause_sensor not in sensor_to_idx:
                continue
            
            cause_idx = sensor_to_idx[cause_sensor]
            
            # Define time window for effects
            effect_start = cause_event['end_time']
            effect_end = effect_start + pd.Timedelta(minutes=time_window_minutes)
            
            # Find candidate effect events
            candidates = events[
                (events['start_time'] >= effect_start) &
                (events['start_time'] <= effect_end)
            ]
            
            # Check each candidate for connectivity
            for _, effect_event in candidates.iterrows():
                effect_sensor = str(effect_event['sensor_id'])
                
                if effect_sensor not in sensor_to_idx or effect_sensor == cause_sensor:
                    continue
                
                effect_idx = sensor_to_idx[effect_sensor]
                conn_strength = connectivity.get((cause_idx, effect_idx), 0.0)
                
                # Record event if connectivity greater than threshold
                if conn_strength > min_connectivity:
                    time_delay = (effect_event['start_time'] - cause_event['end_time']).total_seconds() / 60
                    
                    causal_relationships.append({
                        'cause_event_id': cause_event['event_id'],
                        'effect_event_id': effect_event['event_id'],
                        'cause_sensor': cause_sensor,
                        'effect_sensor': effect_sensor,
                        'cause_start': cause_event['start_time'],
                        'cause_end': cause_event['end_time'],
                        'effect_start': effect_event['start_time'],
                        'effect_end': effect_event['end_time'],
                        'time_delay_minutes': time_delay,
                        'connectivity_strength': conn_strength,
                        'cause_severity': cause_event.get('severity', 'unknown'),
                        'effect_severity': effect_event.get('severity', 'unknown')
                    })
        
        return pd.DataFrame(causal_relationships)

    @staticmethod
    def analyze_causal_chains(causal_df: pd.DataFrame, max_depth: int = 5) -> List[List[int]]:
        """Identifies chains of causally related events (depth first search to find cascading congestion)."""
        
        if causal_df.empty:
            return []
        
        # Build causality graph
        causality_graph = {}
        for _, row in causal_df.iterrows():
            
            cause = row['cause_event_id']
            effect = row['effect_event_id']

            if cause not in causality_graph:
                causality_graph[cause] = []

            causality_graph[cause].append(effect)
        
        # Find chains using DFS
        chains = []
        
        def dfs(event_id: int, current_chain: List[int], visited: set):
            if len(current_chain) >= max_depth:
                return
            
            if event_id in causality_graph:
                for next_event in causality_graph[event_id]:
                    if next_event not in visited:

                        new_chain = current_chain + [next_event]
                        chains.append(new_chain)
                        dfs(next_event, new_chain, visited | {next_event})
        
        # Start DFS from each event that could be a root cause
        root_events = set(causal_df['cause_event_id']) - set(causal_df['effect_event_id'])
        
        for root in root_events:
            chains.append([root])
            dfs(root, [root], {root})
        
        # Sort by chain length (longest first)
        chains.sort(key=len, reverse=True)
        
        return chains

    @staticmethod
    def summarize_causal_patterns(causal_df: pd.DataFrame) -> None:
        """Print summary statistics about causal relationships."""
        
        if causal_df.empty:
            print("No causal relationships found.")
            return

        print(f"\nTotal causal relationships identified: {len(causal_df)}")

        print(f"\nTop 10 Most Causally Active Sensors (as cause):")
        cause_counts = causal_df['cause_sensor'].value_counts().head(10)
        for sensor, count in cause_counts.items():
            print(f"  Sensor {sensor}: {count} downstream events")
        
        print(f"\nTop 10 Most Affected Sensors (as effect):")
        effect_counts = causal_df['effect_sensor'].value_counts().head(10)
        for sensor, count in effect_counts.items():
            print(f"  Sensor {sensor}: {count} upstream causes")
        
        print(f"\nSeverity Propagation:")
        severity_pairs = causal_df.groupby(['cause_severity', 'effect_severity']).size()
        print(severity_pairs)