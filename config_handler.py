"""
config_handler.py

Handles configuration files and evolutionary checkpoint serialization.
Supports SISO, MISO, SIMO, and MIMO system identification modes.
"""

import json
import pickle
import os
import time


def atomic_replace_retry(src_path, dst_path, retries=5, delay=0.05):
    # Windows Defender or Indexer can briefly hold read handles on checkpoint files.
    # Retry a few times so training doesn't crash with PermissionError [WinError 5].
    for attempt in range(retries):
        try:
            os.replace(src_path, dst_path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))

DEFAULT_HYPERPARAMETERS = {
    "model_type": "ARMAX",
    "system_mode": "SISO",
    "input_col_idx": 0,
    "output_col_idx": 0,
    "na": 2,
    "nb": 2,
    "nc": 2,
    "nk": 0,
    "mlp_hidden_nodes": [5],
    "pop_size": 150,
    "num_gen": 100,
    "crossover_prob": 0.1,
    "mutation_sigma": 0.5,
    "mutation_type": "Gaussian",
    "selection_type": "Tournament",
    "tournament_size": 4,
    "checkpoint_freq": 10,
    "k_folds": 10,
    "random_seed": 42,
    "early_stop_patience": 50,
    "early_stop_tol": 1e-6,
    "early_stop_rel_tol": 1e-4,
    "lamarckian_refine_freq": 5,
    "refine_top_candidates": 6,
    "l2_reg": 0.0001,
    "crossover_alpha": 0.25,
    "stability_threshold": 0.98,
    "search_bounds": [-0.5, 0.5],
    "benchmark_dimensions": 5,
    "benchmark_trials": 10,
    "auto_clear_monitor": False
}

PARAMETER_LABELS = {
    "model_type": "Model Type",
    "system_mode": "System Mode (IO)",
    "input_col_idx": "Input Channel Index",
    "output_col_idx": "Output Channel Index",
    "na": "AR Lags (na)",
    "nb": "Input Lags (nb)",
    "nc": "Error Lags (nc)",
    "nk": "Time Delay (nk)",
    "mlp_hidden_nodes": "MLP Hidden Neurons",
    "selection_type": "Selection Strategy",
    "tournament_size": "Tournament Competitors (q)",
    "mutation_type": "Mutation Operator",
    "mutation_sigma": "Initial Mutation Step (σ)",
    "pop_size": "Population Size (μ)",
    "crossover_prob": "Crossover Probability (pc)",
    "num_gen": "Max Generations",
    "early_stop_patience": "Early Stop Patience (0=Auto, -1=Off)",
    "k_folds": "Cross-Validation Folds (K)",
    "random_seed": "Random Seed",
    "l2_reg": "L2 Regularization (λ)",
    "checkpoint_freq": "Checkpoint Frequency"
}


def get_parameter_label(key):
    return PARAMETER_LABELS.get(key, key)


def load_hyperparameters(filepath="config.json"):
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(DEFAULT_HYPERPARAMETERS, file, indent=4)
        return DEFAULT_HYPERPARAMETERS.copy()

    with open(filepath, "r", encoding="utf-8") as file:
        parsed_json = json.load(file)

    merged = DEFAULT_HYPERPARAMETERS.copy()
    merged.update(parsed_json)
    merged["system_mode"] = str(merged.get("system_mode", "SISO")).upper().strip()
    return merged


def _generate_architecture_signature(config):
    if not config:
        return None
    model_type = config.get("model_type", "ARMAX")
    sys_mode = str(config.get("system_mode", "SISO")).upper()

    keys = ["model_type", "system_mode", "na", "nb", "nk", "pop_size", "k_folds"]
    if sys_mode in ("SISO", "SIMO"):
        keys.append("input_col_idx")
    if sys_mode in ("SISO", "MISO"):
        keys.append("output_col_idx")

    if model_type == "ARMAX":
        keys.append("nc")
    elif model_type == "MLP":
        keys.append("mlp_hidden_nodes")

    return {k: config.get(k) for k in keys if k in config}


def save_checkpoint_state(population, generation, hall_of_fame, logbook, filepath, config=None, is_completed=False, elapsed_time=0.0):
    state_payload = {
        "population": population,
        "generation": generation,
        "hall_of_fame": hall_of_fame,
        "logbook": logbook,
        "signature": _generate_architecture_signature(config),
        "is_completed": is_completed,
        "elapsed_time": elapsed_time
    }
    tmp_filepath = filepath + ".tmp"
    with open(tmp_filepath, "wb") as state_file:
        pickle.dump(state_payload, state_file)
    atomic_replace_retry(tmp_filepath, filepath)
    print(f"[*] State saved at generation {generation} -> {os.path.basename(filepath)}")


def load_checkpoint_state(filepath, config=None):
    if filepath and os.path.exists(filepath):
        if os.path.getsize(filepath) == 0:
            print(f"[!] Warning: Checkpoint file '{filepath}' is empty (0 bytes). Starting fresh run.")
            return None
        try:
            with open(filepath, "rb") as state_file:
                restored_state = pickle.load(state_file)

            expected_sig = _generate_architecture_signature(config)
            saved_sig = restored_state.get("signature")

            if isinstance(expected_sig, dict) and isinstance(saved_sig, dict):
                core_keys = ("model_type", "system_mode", "na", "nb", "nc", "nk", "mlp_hidden_nodes", "pop_size", "k_folds")
                for k in core_keys:
                    if k in expected_sig and k in saved_sig:
                        val_exp, val_saved = expected_sig[k], saved_sig[k]
                        # Compare values flexibly (e.g. int vs float) so we don't trigger false mismatches
                        match = False
                        if isinstance(val_exp, list) or isinstance(val_saved, list):
                            match = (list(val_exp) == list(val_saved))
                        elif isinstance(val_exp, (int, float)) or isinstance(val_saved, (int, float)):
                            try:
                                match = (int(float(val_exp)) == int(float(val_saved)))
                            except (ValueError, TypeError):
                                match = (str(val_exp).strip() == str(val_saved).strip())
                        else:
                            match = (str(val_exp).upper().strip() == str(val_saved).upper().strip())

                        if not match:
                            print(f"[!] Architecture mismatch on '{k}' ({val_exp} vs {val_saved}) in '{filepath}'. Starting fresh run.")
                            return None

            print(f"[*] Loaded checkpoint from generation {restored_state['generation']}.")
            return restored_state
        except Exception as e:
            print(f"[!] Could not read checkpoint '{filepath}' ({e}). Starting fresh run.")
            return None
    return None