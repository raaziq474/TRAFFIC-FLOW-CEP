from pathlib import Path
import pandas as pd
import pickle

#################
# Dataset downloader and dataloader module
#################


DEFAULT_DATA_DIR = Path("data")

def _get_file_path(filename: str, format: str = "csv", data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """
    Helper to build file path based on dataset name, format, and data directory.
    Automatically avoids double extensions if dataset_name already contains it.
    """
    if format not in {"csv", "pkl"}:
        raise ValueError("format must be 'csv' or 'pkl'")

    data_dir.mkdir(parents=True, exist_ok=True)

    path = data_dir / filename

    # Add extension only if it's missing
    if path.suffix != f".{format}":
        path = path.with_suffix(f".{format}")

    return path


def download_dataset(dataset_name: str, format: str = "csv", data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """
    Downloads and saves the dataset as CSV or PKL to the specified data_dir, defualt is 'data'.
    Requires tsl, but only imports it when called.
    """

    from tsl.datasets import MetrLA, PemsBay

    if dataset_name == "la":
        dataset = MetrLA(impute_zeros=True)
    elif dataset_name == "bay":
        dataset = PemsBay()
    elif dataset_name == "ne-bj":
        raise NotImplementedError
    else:
        raise ValueError("Downloadable datasets are: 'la' or 'bay'")

    df = dataset.target

    # Pre-process sensor IDs (0, 77123) -> 771123
    if isinstance(df.columns[0], tuple):
        df.columns = [c[0] for c in df.columns]

    dataset_path = _get_file_path(filename=dataset_name, format=format, data_dir=data_dir)

    # Save dataset 
    if format == "csv":
        df.to_csv(dataset_path, index=True)
    elif format == "pkl":
        df.to_pickle(dataset_path)
    else:
        raise Exception("Downloadable formats are 'csv' and 'pkl'")
    print("Downloaded Dataset")

    # Save distance predefined adjacency matrix, can only be saved as pkl
    adj = dataset.get_connectivity(method="distance", threshold=0.1, include_self=False, normalize_axis=1)
    adj_name = dataset_name + "_adj"
    adj_path = _get_file_path(filename=adj_name, format='pkl', data_dir=data_dir)

    with open(adj_path, "wb") as adj_pkl:
        pickle.dump(adj, adj_pkl)
    print("Downloaded Adjacency Matrix")

    return dataset_path


def load_csv(dataset_name: str, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load a CSV dataset"""

    path = _get_file_path(dataset_name, format="csv", data_dir=data_dir)
    
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run download_dataset('{dataset_name}', format='csv', data_dir='{data_dir}') first."
        )
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_pkl(dataset_name: str, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load a PKL dataset"""

    path = _get_file_path(dataset_name, format="pkl", data_dir=data_dir)
    
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run download_dataset('{dataset_name}', format='pkl', data_dir='{data_dir}') first."
        )
    
    df = pd.read_pickle(path)
    return df


if __name__ == "__main__":
    
    # EXAMPLE USAGE,
    dataset = 'la'

    # Example 1: Download and save as csv
    print("Downloading and saving as csv")
    csv_path = download_dataset(dataset, format="csv")
    print(f"Saved CSV to: {csv_path}\n")

    # Example 2: Download and save as pkl 
    print("\nDownloading and saving as pkl")
    pkl_path = download_dataset(dataset, format="pkl")
    print(f"Saved PKL to: {pkl_path}\n")

    # Example 3: Loading csv dataset
    print("Loading CSV")
    df_csv = load_csv(dataset)
    print("Shape:", df_csv.shape)
    print(df_csv.head())

    # Example 4: Loading PKL dataset
    print("Loading PKL")
    df_pkl = load_pkl(dataset)
    print("Shape:", df_pkl.shape)
    print(df_pkl.head())

    # Example 5: Loading custom csv dataset (predictions from agcrn)
    # On metr-la test partition
    df_csv = load_csv("agcrn_actuals_h2")
    print("Shape:", df_csv.shape)
    print(df_csv.head())

    # Adjacency matrix can be loaded the exact same way
    adj = load_pkl("la_adj.pkl")
    print("Shape:", adj.shape)
    print(adj.head())