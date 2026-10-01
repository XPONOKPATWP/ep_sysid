"""
config_handler.py

Manages optimization parameters and ARMAX configurations.
Handles saving and loading the algorithm state.
"""

import json
import pickle
import os

DEFAULT_HYPERPARAMETERS = {
    "na": 2,
    "nb": 2,
    "nc": 2,
    "nk": 1,
    "pop_size": 150,
    "num_gen": 100,
    "mutation_sigma": 0.5,
    "checkpoint_freq": 10
}

PARAMETER_LABELS = {
    "na": "AR Lags (na)",
    "nb": "Input Lags (nb)",
    "nc": "MA Error Lags (nc)",
    "nk": "Time Delay (nk)",
    "pop_size": "Population Size",
    "num_gen": "Max Generations",
    "mutation_sigma": "Initial Mutation Step Size",
    "checkpoint_freq": "Checkpoint Frequency"
}


def get_parameter_label(key):
    """Maps internal configuration keys to GUI-friendly labels."""
    return PARAMETER_LABELS.get(key, key)


def load_hyperparameters(filepath="config.json"):
    """
    Loads the system configuration from disk, falling back to
    defaults if the file is missing or incomplete.
    """
    if not os.path.exists(filepath):
        with open(filepath, "w") as file:
            json.dump(DEFAULT_HYPERPARAMETERS, file, indent=4)
        return DEFAULT_HYPERPARAMETERS.copy()

    with open(filepath, "r") as file:
        parsed_json = json.load(file)

    active_config = {k: parsed_json.get(k, v) for k, v in DEFAULT_HYPERPARAMETERS.items()}
    return active_config


def _generate_architecture_signature(config):
    """
    Extracts structural hyperparameters to validate checkpoint compatibility.
    """
    if not config:
        return None
    return {k: config.get(k) for k in ("na", "nb", "nc", "nk", "pop_size")}


def save_checkpoint_state(population, generation, hall_of_fame, logbook, filepath, config=None):
    """
    Serializes the current generation's evolutionary state.
    """
    state_payload = {
        "population": population,
        "generation": generation,
        "hall_of_fame": hall_of_fame,
        "logbook": logbook,
        "signature": _generate_architecture_signature(config)
    }
    with open(filepath, "wb") as state_file:
        pickle.dump(state_payload, state_file)
    print(f"[*] State saved at generation {generation}.")


def load_checkpoint_state(filepath, config=None):
    """
    Restores the evolutionary state from a binary checkpoint.
    Validates the configuration signature against the requested architecture
    to avoid dimensionality mismatches during resume.
    """
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, "rb") as state_file:
                restored_state = pickle.load(state_file)

            expected_signature = _generate_architecture_signature(config)
            saved_signature = restored_state.get("signature")

            if expected_signature and saved_signature != expected_signature:
                print(f"[!] Configuration mismatch in '{filepath}'. Starting fresh run.")
                return None

            print(f"[*] Loaded state from Generation {restored_state['generation']}.")
            return restored_state
        except Exception as e:
            print(f"[!] Invalid checkpoint file '{filepath}' ({e}). Starting fresh run.")
            return None
    return None