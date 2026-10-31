"""
multivariable_data_parser.py

Data loader and signal router for SISO, MISO, SIMO, and MIMO system identification.
Handles downloading datasets from the UCI repository, local caching, and CSV parsing.
"""

import os
import pickle
import numpy as np
import pandas as pd

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None

# Dedicated cache filename so old sorted SISO caches are never loaded
LOCAL_DATA_CACHE_FILE = "uci_multivariable_cache.pkl"

# Target column names when a UCI dataset packages all columns inside the features table
PREFERRED_TARGET_COLUMNS = ("TEY", "PE", "Appliances", "Y1", "Y2", "Heating_Load", "Cooling_Load", "CO", "NOx")

SUPPORTED_SYSTEM_MODES = ("SISO", "MISO", "SIMO", "MIMO")


def _extract_multivariable_signals(features_df, targets_df):
    if features_df is None or features_df.empty:
        raise ValueError("Dataset error: Missing feature data (X).")

    numeric_features = features_df.apply(lambda col: pd.to_numeric(col, errors="coerce")).dropna(axis=1, how="all")

    # If the repository put targets inside features_df, extract the target columns
    if targets_df is None or targets_df.empty:
        if numeric_features.shape[1] < 2:
            raise ValueError("Dataset error: Missing target data (y) and fewer than 2 numeric columns.")

        matched_targets = [col for col in PREFERRED_TARGET_COLUMNS if col in numeric_features.columns]
        if matched_targets:
            numeric_targets = numeric_features[matched_targets].copy()
            numeric_features = numeric_features.drop(columns=matched_targets)
        else:
            last_col = numeric_features.columns[-1]
            numeric_targets = numeric_features[[last_col]].copy()
            numeric_features = numeric_features.drop(columns=[last_col])
    else:
        numeric_targets = targets_df.apply(lambda col: pd.to_numeric(col, errors="coerce")).dropna(axis=1, how="all")

    if numeric_features.empty or numeric_targets.empty:
        raise ValueError("No valid numeric columns found after cleaning.")

    # Keep original time-series row order; do not sort by input so dynamic relationships stay intact
    merged_df = pd.concat([numeric_features, numeric_targets], axis=1).dropna().reset_index(drop=True)
    if merged_df.empty:
        raise ValueError("Dataset is empty after dropping missing values.")

    num_u = numeric_features.shape[1]
    u_matrix = merged_df.iloc[:, :num_u].values.astype(float)
    y_matrix = merged_df.iloc[:, num_u:].values.astype(float)

    return u_matrix, y_matrix


def slice_signals_by_system_mode(u_raw, y_raw, system_mode="SISO", input_col_idx=0, output_col_idx=0):
    u_mat = np.asarray(u_raw, dtype=float)
    y_mat = np.asarray(y_raw, dtype=float)

    if u_mat.ndim == 1:
        u_mat = u_mat.reshape(-1, 1)
    if y_mat.ndim == 1:
        y_mat = y_mat.reshape(-1, 1)

    mode = str(system_mode).upper().strip()
    if mode not in SUPPORTED_SYSTEM_MODES:
        mode = "SISO"

    safe_in = int(input_col_idx) if 0 <= int(input_col_idx) < u_mat.shape[1] else 0
    safe_out = int(output_col_idx) if 0 <= int(output_col_idx) < y_mat.shape[1] else 0

    if mode == "SISO":
        u_selected = u_mat[:, [safe_in]]
        y_selected = y_mat[:, [safe_out]]
    elif mode == "MISO":
        u_selected = u_mat
        y_selected = y_mat[:, [safe_out]]
    elif mode == "SIMO":
        u_selected = u_mat[:, [safe_in]]
        y_selected = y_mat
    else:  # MIMO
        u_selected = u_mat
        y_selected = y_mat

    u_final = u_selected.flatten() if u_selected.shape[1] == 1 else u_selected
    y_final = y_selected.flatten() if y_selected.shape[1] == 1 else y_selected

    return u_final, y_final


def fetch_and_parse_uci_dataset(uci_repository_id):
    # Downloads dataset by UCI ID and extracts full 2D multivariable matrices
    try:
        if fetch_ucirepo is None:
            raise ImportError("Package 'ucimlrepo' is not installed.")

        repo_id = int(uci_repository_id)
        print(f"Fetching UCI Dataset ID: {repo_id}...")
        dataset_payload = fetch_ucirepo(id=repo_id)

        raw_name = dataset_payload.metadata.name or f"UCI_{repo_id}"
        dataset_name = raw_name.strip().replace(" ", "_").replace("-", "_")

        u_matrix, y_matrix = _extract_multivariable_signals(
            dataset_payload.data.features, dataset_payload.data.targets
        )
        return dataset_name, u_matrix, y_matrix, None
    except Exception as error:
        error_msg = str(error)
        print(f"Failed to fetch ID {uci_repository_id}: {error_msg}")
        return None, None, None, error_msg


def load_cached_uci_benchmarks():
    # Loads default benchmark datasets from cache, or downloads them if cache is missing
    if os.path.exists(LOCAL_DATA_CACHE_FILE):
        try:
            print(f"Loading datasets from cache ({LOCAL_DATA_CACHE_FILE})...")
            with open(LOCAL_DATA_CACHE_FILE, "rb") as cache_file:
                cached_data = pickle.load(cache_file)
            if isinstance(cached_data, dict) and cached_data:
                return cached_data
        except Exception as error:
            print(f"Cache read failed ({error}). Re-fetching...")

    _, u_power, y_power, _ = fetch_and_parse_uci_dataset(294)
    _, u_energy, y_energy, _ = fetch_and_parse_uci_dataset(242)

    benchmark_datasets = {}
    if u_energy is not None and y_energy is not None:
        benchmark_datasets["UCI_Energy_Efficiency"] = (u_energy, y_energy)
    if u_power is not None and y_power is not None:
        benchmark_datasets["UCI_Power_Plant"] = (u_power, y_power)

    if benchmark_datasets:
        try:
            with open(LOCAL_DATA_CACHE_FILE, "wb") as cache_file:
                pickle.dump(benchmark_datasets, cache_file)
            print(f"Saved multivariable datasets to {LOCAL_DATA_CACHE_FILE}.")
        except Exception as error:
            print(f"Warning: Could not write cache file: {error}")

    return benchmark_datasets


def parse_local_csv_file(filepath, num_outputs=1):
    # Reads a custom CSV file; last num_outputs columns are treated as target outputs Y
    try:
        print(f"Reading CSV: {filepath}...")
        raw_df = pd.read_csv(filepath)
        if raw_df.shape[1] < 2:
            raise ValueError("CSV requires at least 2 columns (Inputs and Outputs).")

        n_out = max(1, min(int(num_outputs), raw_df.shape[1] - 1))
        features_df = raw_df.iloc[:, :-n_out]
        targets_df = raw_df.iloc[:, -n_out:]
        u_matrix, y_matrix = _extract_multivariable_signals(features_df, targets_df)

        base_name = os.path.splitext(os.path.basename(filepath))[0].replace(" ", "_")
        return base_name, u_matrix, y_matrix, None
    except Exception as error:
        error_msg = str(error)
        print(f"CSV read failed for '{filepath}': {error_msg}")
        return None, None, None, error_msg


def get_dataset_summary_info(dataset_name, loaded_datasets=None):
    # Returns descriptive statistics and channel counts for summary workbooks
    if loaded_datasets and dataset_name in loaded_datasets:
        u_sig, y_sig = loaded_datasets[dataset_name]
        if u_sig is not None and y_sig is not None and len(y_sig) > 0:
            u_arr = np.asarray(u_sig, dtype=float)
            y_arr = np.asarray(y_sig, dtype=float)
            n_u = 1 if u_arr.ndim == 1 else u_arr.shape[1]
            n_y = 1 if y_arr.ndim == 1 else y_arr.shape[1]
            return {
                "Dataset_Samples": int(len(y_arr)),
                "IO_Channels": f"{n_u} In / {n_y} Out",
                "Input_Range": f"[{np.min(u_arr):.2f}, {np.max(u_arr):.2f}]",
                "Output_Mean_Std": f"{np.mean(y_arr):.2f} +/- {np.std(y_arr):.2f}",
                "Output_Min_Max": f"[{np.min(y_arr):.2f}, {np.max(y_arr):.2f}]"
            }
    return {
        "Dataset_Samples": "",
        "IO_Channels": "",
        "Input_Range": "",
        "Output_Mean_Std": "",
        "Output_Min_Max": ""
    }