"""
siso_data_parser.py

Data loader for Single-Input Single-Output (SISO) datasets.
Handles retrieving and caching time-series data.
"""

import os
import pickle
import pandas as pd
from ucimlrepo import fetch_ucirepo

LOCAL_DATA_CACHE_FILE = "uci_default_cache.pkl"


def _extract_siso_signals(features_df, targets_df, input_col_idx=0, output_col_idx=0):
    """
    Extracts contiguous numeric signals from raw DataFrames.
    """
    if features_df is None or features_df.empty:
        raise ValueError("Dataset error: Missing feature data (X).")
    if targets_df is None or targets_df.empty:
        raise ValueError("Dataset error: Missing target data (y).")

    # Coerce invalid string entries to NaN and drop completely empty columns
    numeric_features = features_df.apply(lambda col: pd.to_numeric(col, errors='coerce')).dropna(axis=1, how='all')
    numeric_targets = targets_df.apply(lambda col: pd.to_numeric(col, errors='coerce')).dropna(axis=1, how='all')

    if numeric_features.empty or numeric_targets.empty:
        raise ValueError("No valid numeric columns found after processing.")

    # Inner join on index and drop any rows containing NaN to ensure a continuous time-series
    merged_dataframe = pd.concat([numeric_features, numeric_targets], axis=1).dropna()
    if merged_dataframe.empty:
        raise ValueError("Dataset is empty after dropping missing values.")

    num_feature_cols = numeric_features.shape[1]

    safe_input_idx = input_col_idx if input_col_idx < num_feature_cols else 0
    u_signal = merged_dataframe.iloc[:, safe_input_idx].values.astype(float)

    clean_targets = merged_dataframe.iloc[:, num_feature_cols:]
    safe_output_idx = output_col_idx if output_col_idx < clean_targets.shape[1] else 0
    y_signal = clean_targets.iloc[:, safe_output_idx].values.astype(float)

    return u_signal, y_signal


def fetch_and_parse_uci_dataset(uci_repository_id):
    """
    Retrieves and parses datasets directly from the UCI API.
    """
    try:
        print(f"Fetching Dataset ID: {uci_repository_id}...")
        dataset_payload = fetch_ucirepo(id=int(uci_repository_id))
        dataset_name = dataset_payload.metadata.name.replace(" ", "_")
        u_signal, y_signal = _extract_siso_signals(dataset_payload.data.features, dataset_payload.data.targets)
        return dataset_name, u_signal, y_signal, None
    except Exception as error:
        error_msg = str(error)
        print(f"Failed to fetch ID {uci_repository_id}: {error_msg}")
        return None, None, None, error_msg


def load_cached_uci_benchmarks():
    """
    Loads baseline system identification datasets from a local cache,
    fetching them if the cache does not exist.
    """
    if os.path.exists(LOCAL_DATA_CACHE_FILE):
        try:
            print(f"Loading datasets from local file ({LOCAL_DATA_CACHE_FILE})...")
            with open(LOCAL_DATA_CACHE_FILE, "rb") as cache_file:
                return pickle.load(cache_file)
        except Exception as error:
            print(f"Failed to read local cache ({error}). Re-fetching...")

    # Fetch default fallback datasets (e.g. CCPP and Energy Efficiency)
    _, u_power, y_power, _ = fetch_and_parse_uci_dataset(294)
    _, u_energy, y_energy, _ = fetch_and_parse_uci_dataset(242)

    benchmark_datasets = {
        "UCI_Power_Plant": (u_power, y_power),
        "UCI_Energy_Efficiency": (u_energy, y_energy)
    }

    try:
        with open(LOCAL_DATA_CACHE_FILE, "wb") as cache_file:
            pickle.dump(benchmark_datasets, cache_file)
        print(f"Saved datasets to local file ({LOCAL_DATA_CACHE_FILE}).")
    except Exception as error:
        print(f"Warning: Could not save to local file: {error}")

    return benchmark_datasets


def parse_local_csv_file(filepath):
    """
    Parses custom SISO time-series data from a local CSV.
    Assumes the final column is the target y(t) and earlier columns are u(t).
    """
    try:
        print(f"Reading CSV: {filepath}...")
        raw_dataframe = pd.read_csv(filepath)
        if raw_dataframe.shape[1] < 2:
            raise ValueError("CSV needs at least 2 columns for Input and Output.")

        features_df = raw_dataframe.iloc[:, :-1]
        targets_df = raw_dataframe.iloc[:, -1:]
        u_signal, y_signal = _extract_siso_signals(features_df, targets_df)

        base_filename = os.path.splitext(os.path.basename(filepath))[0].replace(" ", "_")
        return base_filename, u_signal, y_signal, None
    except Exception as error:
        error_msg = str(error)
        print(f"CSV read failed for '{filepath}': {error_msg}")
        return None, None, None, error_msg