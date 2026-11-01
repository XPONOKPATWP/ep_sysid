"""
ep_algorithm_core.py

Core evolutionary algorithm module using DEAP for system identification.
Supports SISO, MISO, SIMO, and MIMO systems for ARMAX, MLP, and SVR models.
Includes multiprocessing workers, self-adaptive mutations, round-robin tournament selection,
and stability checks.
"""

import os
# Restrict NumPy/OpenBLAS to 1 thread per worker process so np.dot in MLP
# does not spawn dozens of threads and starve the CustomTkinter GUI
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import time
import numpy as np
import random
import multiprocessing
from deap import base, creator, tools

from mlp_simulator import simulate_mlp_system, get_mlp_parameter_count, build_mlp_feature_matrix
from armax_simulator import simulate_armax_system, get_armax_parameter_count, build_armax_feature_cache
from svr_simulator import simulate_svr_system, get_svr_parameter_count, build_svr_feature_cache
from validation_metrics import compute_mse
from config_handler import save_checkpoint_state, load_checkpoint_state

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if not hasattr(creator, "EPIndividual"):
    creator.create("EPIndividual", list, fitness=creator.FitnessMin, strategy=None)

# Global worker variables populated once per fold via Pool initializer
_WORKER_U = None
_WORKER_Y = None
_WORKER_HPARAMS = None
_WORKER_CACHE = None


def _init_worker_process(u_data, y_data, hparams):
    # Store dataset and lag matrices in global worker memory once
    # so multiprocessing doesn't waste time serializing large arrays for each evaluation
    global _WORKER_U, _WORKER_Y, _WORKER_HPARAMS, _WORKER_CACHE
    _WORKER_U = u_data
    _WORKER_Y = y_data
    _WORKER_HPARAMS = hparams
    _WORKER_CACHE = None

    if u_data is not None and y_data is not None and hparams:
        model_type = hparams.get("model_type", "MLP")
        na, nb, nk = int(hparams["na"]), int(hparams["nb"]), int(hparams["nk"])
        if model_type == "MLP":
            _WORKER_CACHE = build_mlp_feature_matrix(u_data, y_data, na, nb, nk)
        elif model_type == "ARMAX":
            nc = int(hparams.get("nc", 2))
            _WORKER_CACHE = build_armax_feature_cache(u_data, y_data, na, nb, nc, nk)
        elif model_type == "SVR":
            _WORKER_CACHE = build_svr_feature_cache(u_data, y_data, na, nb, nk)


def _evaluate_worker_genome(genome):
    return evaluate_genome_fitness(genome, _WORKER_U, _WORKER_Y, _WORKER_HPARAMS, precomputed_features=_WORKER_CACHE)


def initialize_ep_genome(genome_class, parameter_count, min_bound=-0.5, max_bound=0.5, initial_sigma=0.05, layer_bounds=None):
    if layer_bounds is not None and len(layer_bounds) == parameter_count:
        b_arr = np.asarray(layer_bounds, dtype=float)
        weights = np.random.uniform(-b_arr, b_arr)
        genome = genome_class(weights.tolist())
        base_span = max(1e-3, 0.5 * (max_bound - min_bound))
        sigmas = np.clip(initial_sigma * (b_arr / base_span), 1e-4, 2.0)
        genome.strategy = sigmas.tolist()
    else:
        genome = genome_class(np.random.uniform(min_bound, max_bound, parameter_count))
        genome.strategy = [initial_sigma for _ in range(parameter_count)]
    return genome


def _build_mlp_xavier_bounds(na, nb, hidden_units, n_u=1, n_y=1, max_scale=0.5):
    # Computes per-parameter Xavier/Glorot bounds sqrt(6 / (fan_in + fan_out)) so high-lag MLPs never saturate tanh
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]
    n_in = (int(na) * int(n_y)) + (int(nb) * int(n_u))
    bounds = []
    layer_in = n_in
    for units in hidden_units:
        limit = min(max_scale, float(np.sqrt(6.0 / max(1, layer_in + units))))
        bounds.extend([limit] * (layer_in * units))
        bounds.extend([min(0.1, limit)] * units)
        layer_in = units
    out_limit = min(max_scale, float(np.sqrt(6.0 / max(1, layer_in + n_y))))
    bounds.extend([out_limit] * (layer_in * n_y))
    bounds.extend([min(0.1, out_limit)] * n_y)
    return bounds


def _build_armax_scaled_bounds(na, nb, nc, n_u=1, n_y=1, max_scale=0.5):
    """
    Scale initial bounds based on the number of lags.
    Keeping the sum of coefficients small ensures the initial polynomials
    start off stable inside the unit circle.
    """
    na, nb, nc = int(na), int(nb), int(nc)
    n_u, n_y = int(n_u), int(n_y)
    ar_limit = min(max_scale, 0.85 / max(1, na))
    b_limit = min(max_scale, float(np.sqrt(3.0 / max(1, nb * n_u))))
    ma_limit = min(max_scale, 0.85 / max(1, nc))

    per_output = ([ar_limit] * na) + ([b_limit] * (nb * n_u)) + ([ma_limit] * nc)
    return per_output * n_y


def fast_clone_genome(genome):
    # Standard copy.deepcopy is too slow for 150 individuals x 100 generations;
    # a shallow copy of the list and strategy avoids this bottleneck
    cloned = creator.EPIndividual(genome)
    cloned.strategy = list(genome.strategy) if getattr(genome, "strategy", None) is not None else [0.05] * len(genome)
    if genome.fitness.valid:
        cloned.fitness.values = genome.fitness.values
    return cloned


def sel_ep_round_robin_tournament(individuals, k, tournsize=4):
    """
    Round-robin tournament selection: each individual competes against q random
    opponents from the pool and counts wins based on fitness. The top k survive.
    """
    n_pool = len(individuals)
    if n_pool <= k:
        return [fast_clone_genome(ind) for ind in individuals]

    fitnesses = np.array([ind.fitness.values[0] for ind in individuals], dtype=float)
    q = min(max(1, int(tournsize)), n_pool - 1)
    # Offset by 1..n_pool-1 so an individual never competes against itself
    offsets = np.random.randint(1, n_pool, size=(n_pool, q))
    opp_indices = (np.arange(n_pool)[:, None] + offsets) % n_pool
    wins = np.sum(fitnesses[:, None] <= fitnesses[opp_indices], axis=1)

    order = np.lexsort((fitnesses, -wins))
    selected_indices = order[:k]
    return [fast_clone_genome(individuals[idx]) for idx in selected_indices]


def apply_blend_crossover_with_strategy(ind1, ind2, alpha=0.25):
    # Vectorized BLX-alpha crossover for weights and randomized convex blending for strategy step-sizes
    n = len(ind1)
    w1 = np.asarray(ind1, dtype=float)
    w2 = np.asarray(ind2, dtype=float)
    gamma = (1.0 + 2.0 * alpha) * np.random.random(n) - alpha

    ind1[:] = ((1.0 - gamma) * w1 + gamma * w2).tolist()
    ind2[:] = (gamma * w1 + (1.0 - gamma) * w2).tolist()

    s1_raw = getattr(ind1, "strategy", None)
    s2_raw = getattr(ind2, "strategy", None)
    s1 = np.asarray(s1_raw if (s1_raw is not None and len(s1_raw) == n) else [0.05] * n, dtype=float)
    s2 = np.asarray(s2_raw if (s2_raw is not None and len(s2_raw) == n) else [0.05] * n, dtype=float)

    beta = np.random.random(n)
    s_child1 = np.clip((1.0 - beta) * s1 + beta * s2, 1e-5, 10.0).tolist()
    s_child2 = np.clip(beta * s1 + (1.0 - beta) * s2, 1e-5, 10.0).tolist()

    try:
        ind1.strategy = s_child1
    except AttributeError:
        pass
    try:
        ind2.strategy = s_child2
    except AttributeError:
        pass
    return ind1, ind2


def apply_self_adaptive_mutation(genome, mutation_type="gaussian"):
    """
    Self-adaptive mutation: mutation step sizes (strategy parameters) evolve
    along with candidate weights. Supports gaussian, cauchy, or hybrid perturbations.
    """
    dimensions = max(1, len(genome))
    # Self-adaptive mutation rates: tau_prime scales global step size (1/sqrt(2D)), tau perturbs each gene (1/sqrt(2*sqrt(D)))
    tau_prime = 1.0 / np.sqrt(2.0 * dimensions)
    tau = 1.0 / np.sqrt(2.0 * np.sqrt(dimensions))

    strat = getattr(genome, "strategy", None)
    if strat is None or len(strat) != dimensions:
        strat_arr = np.full(dimensions, 0.05, dtype=float)
    else:
        strat_arr = np.asarray(strat, dtype=float)

    exponent = np.clip(np.random.normal(0.0, tau_prime) + np.random.normal(0.0, tau, dimensions), -20.0, 20.0)
    s_new = np.clip(strat_arr * np.exp(exponent), 1e-5, 10.0)

    m_type = str(mutation_type).lower().strip()
    if m_type == "cauchy":
        cauchy_step = np.clip(np.random.standard_cauchy(dimensions), -50.0, 50.0)
        delta = s_new * cauchy_step
    elif m_type == "hybrid":
        cauchy_step = np.clip(np.random.standard_cauchy(dimensions), -50.0, 50.0)
        gauss_step = np.random.normal(0.0, 1.0, dimensions)
        delta = s_new * (0.5 * gauss_step + 0.5 * cauchy_step)
    else:
        delta = np.random.normal(0.0, s_new)

    try:
        genome.strategy = s_new.tolist()
    except AttributeError:
        pass
    # Clamp mutated weights to [-1e4, 1e4] so repeated heavy-tailed Cauchy steps
    # cannot drift past float64 squaring limits during L2 regularization
    genome[:] = np.clip(np.asarray(genome, dtype=float) + delta, -1e4, 1e4).tolist()
    return genome,


def apply_self_adaptive_gaussian_mutation(genome):
    # Standard Gaussian self-adaptive mutation
    return apply_self_adaptive_mutation(genome, mutation_type="gaussian")


def _max_root_magnitude_fast(poly_tail, stab_threshold):
    # Quick check: if the sum of absolute coefficients is small enough,
    # all roots are strictly inside the threshold so we can skip np.roots.
    n = len(poly_tail)
    if n == 0:
        return 0.0
    if n == 1:
        return float(abs(poly_tail[0]))
    inv_r = stab_threshold ** (-np.arange(1, n + 1))
    if float(np.dot(np.abs(poly_tail), inv_r)) <= 1.0:
        return 0.0
    try:
        roots = np.roots(np.concatenate([[1.0], poly_tail]))
        return float(np.max(np.abs(roots))) if len(roots) > 0 else 0.0
    except Exception:
        return 2.0


def project_poly_to_stable_disc(poly_tail, max_radius=0.98):
    # If any roots lie outside the unit circle, reflect/scale them back inside |z| <= max_radius
    poly_arr = np.asarray(poly_tail, dtype=float)
    if len(poly_arr) == 0 or _max_root_magnitude_fast(poly_arr, max_radius) <= max_radius:
        return poly_arr
    try:
        roots = np.roots(np.concatenate([[1.0], poly_arr]))
        mags = np.abs(roots)
        unstable = mags > max_radius
        if np.any(unstable):
            roots[unstable] = (roots[unstable] / mags[unstable]) * max_radius
            stabilized_poly = np.real(np.poly(roots))[1:]
            return stabilized_poly
    except Exception:
        pass
    return poly_arr


def evaluate_genome_fitness(genome, u_signal, y_signal, hyperparameters, precomputed_features=None):
    # Simulates the chosen model with candidate parameters and returns regularized MSE
    model_type = hyperparameters.get("model_type", "MLP")
    na, nb, nk = hyperparameters["na"], hyperparameters["nb"], hyperparameters["nk"]

    if model_type == "ARMAX":
        nc = hyperparameters.get("nc", 2)
        predicted_y, max_delay_lag = simulate_armax_system(
            u_signal, y_signal, genome, na, nb, nc, nk, precomputed_cache=precomputed_features
        )
    elif model_type == "SVR":
        predicted_y, max_delay_lag = simulate_svr_system(
            u_signal, y_signal, genome, na, nb, nk, precomputed_cache=precomputed_features
        )
    else:
        hidden_units = hyperparameters.get("mlp_hidden_nodes", [5])
        predicted_y, max_delay_lag = simulate_mlp_system(
            u_signal, y_signal, genome, na, nb, nk, hidden_units, precomputed_features=precomputed_features
        )

    # Calculate MSE on the output signals after the initial lag window.
    # Note: for SVR during training, simulate_svr_system bakes the holdout validation residual
    # directly into predicted_y so this gives pure out-of-sample error.
    mse_score = compute_mse(y_signal[max_delay_lag:], predicted_y[max_delay_lag:])

    if not np.isfinite(mse_score) or mse_score >= 1e6:
        return (1e6,)

    # Penalize unstable AR or MA polynomials so the model stays stable
    if model_type == "ARMAX":
        theta_arr = np.asarray(genome, dtype=float).flatten()
        n_u = 1 if u_signal.ndim == 1 else u_signal.shape[1]
        n_y = 1 if y_signal.ndim == 1 else y_signal.shape[1]
        nc_val = int(hyperparameters.get("nc", 2))
        stride = int(na) + int(nb) * n_u + nc_val
        stab_threshold = float(hyperparameters.get("stability_threshold", 0.98))

        max_p_mag = 0.0
        max_c_mag = 0.0
        for ch in range(n_y):
            offset = ch * stride
            if na > 0:
                ar_slice = theta_arr[offset: offset + na]
                if len(ar_slice) == na:
                    max_p_mag = max(max_p_mag, _max_root_magnitude_fast(ar_slice, stab_threshold))
            if nc_val > 0:
                ma_offset = offset + na + int(nb) * n_u
                ma_slice = theta_arr[ma_offset: ma_offset + nc_val]
                if len(ma_slice) == nc_val:
                    max_c_mag = max(max_c_mag, _max_root_magnitude_fast(ma_slice, stab_threshold))

        if max_p_mag > stab_threshold:
            mse_score += 1e2 * float((max_p_mag - stab_threshold) ** 2)
        if max_c_mag > stab_threshold:
            mse_score += 1e2 * float((max_c_mag - stab_threshold) ** 2)

    # L2 weight decay for ARMAX and MLP to prevent extreme weights and overfitting
    if model_type in ("ARMAX", "MLP"):
        l2_lambda = float(hyperparameters.get("l2_reg", 1e-4))
        with np.errstate(over="ignore", invalid="ignore"):
            l2_penalty = l2_lambda * float(np.mean(np.square(genome)))
        if not np.isfinite(l2_penalty):
            return (1e6,)
        mse_score += l2_penalty

    if not np.isfinite(mse_score) or mse_score >= 1e6:
        return (1e6,)

    return (mse_score,)


def refine_best_genome_lamarckian(best_genome, u_train, y_train, hyperparameters, precomputed_mlp_features=None):
    """
    Optional hybrid step for MLP:
    Uses the hidden layer outputs as basis features and solves Ridge regression
    to optimize the linear output layer directly. If the new weights are better,
    we keep them.
    """
    model_type = hyperparameters.get("model_type", "MLP")
    if model_type != "MLP" or not best_genome:
        return best_genome

    try:
        na, nb, nk = int(hyperparameters["na"]), int(hyperparameters["nb"]), int(hyperparameters["nk"])
        hidden_units = hyperparameters.get("mlp_hidden_nodes", [5])
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]

        if precomputed_mlp_features is not None:
            features, max_delay_lag, _, n_u, n_y, _ = precomputed_mlp_features
        else:
            features, max_delay_lag, _, n_u, n_y, _ = build_mlp_feature_matrix(u_train, y_train, na, nb, nk)

        y_mat = np.asarray(y_train, dtype=float)
        if y_mat.ndim == 1:
            y_mat = y_mat.reshape(-1, 1)
        y_target = y_mat[max_delay_lag:, :]

        theta = np.asarray(best_genome, dtype=float).flatten().copy()
        n_in = (na * n_y) + (nb * n_u)
        idx = 0
        layer_in = n_in
        activations = features

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for units in hidden_units:
                w = theta[idx: idx + layer_in * units].reshape(layer_in, units)
                idx += layer_in * units
                b = theta[idx: idx + units]
                idx += units
                activations = np.tanh(np.dot(activations, w) + b)
                layer_in = units

        ones_col = np.ones((activations.shape[0], 1), dtype=float)
        h_aug = np.hstack([activations, ones_col])
        l2_lambda = float(hyperparameters.get("l2_reg", 1e-4))
        l2_reg = max(1e-4, l2_lambda * float(len(y_target) * n_y) / max(1, len(theta)))
        reg_eye = l2_reg * np.eye(h_aug.shape[1], dtype=float)
        # Keep regularization tiny on the last column (output bias) so we don't pull the DC offset toward zero
        reg_eye[-1, -1] = 1e-6

        gram_mat = np.dot(h_aug.T, h_aug) + reg_eye
        rhs_vec = np.dot(h_aug.T, y_target)
        try:
            wb_opt = np.linalg.solve(gram_mat, rhs_vec)
        except np.linalg.LinAlgError:
            wb_opt, _, _, _ = np.linalg.lstsq(gram_mat, rhs_vec, rcond=None)
        w_out_new = wb_opt[:-1, :].flatten()
        b_out_new = wb_opt[-1, :].flatten()

        out_start = idx
        out_end = idx + (layer_in * n_y) + n_y
        theta[idx: idx + layer_in * n_y] = w_out_new
        theta[idx + layer_in * n_y: out_end] = b_out_new

        cand = fast_clone_genome(best_genome)
        cand[:] = theta.tolist()
        if getattr(cand, "strategy", None) is not None and len(cand.strategy) >= out_end:
            for s_idx in range(out_start, out_end):
                cand.strategy[s_idx] = min(max(1e-4, cand.strategy[s_idx] * 0.25), 0.02)

        new_score = evaluate_genome_fitness(cand, u_train, y_train, hyperparameters, precomputed_features=precomputed_mlp_features)
        old_score = best_genome.fitness.values if best_genome.fitness.valid else evaluate_genome_fitness(
            best_genome, u_train, y_train, hyperparameters, precomputed_features=precomputed_mlp_features
        )
        if new_score[0] < old_score[0]:
            cand.fitness.values = new_score
            return cand
    except Exception:
        pass
    return best_genome


def get_optimal_worker_count(hyperparameters=None, pop_size=None):
    """
    Chooses how many worker processes to spawn based on CPU count.
    Leaves a couple of cores free so the GUI and system stay responsive.
    """
    if hyperparameters and "num_workers" in hyperparameters:
        custom_workers = hyperparameters.get("num_workers")
        if custom_workers is not None and int(custom_workers) > 0:
            return int(custom_workers)

    detected_cores = os.cpu_count() or 4
    usable_cores = max(1, detected_cores - 2)

    active_jobs = max(1, int(hyperparameters.get("active_jobs", 1))) if hyperparameters else 1
    per_job_workers = max(1, usable_cores // active_jobs)

    if pop_size is not None and pop_size > 0:
        per_job_workers = min(per_job_workers, int(pop_size))
    return per_job_workers


def configure_ep_toolbox(hyperparameters, u_sample=None, y_sample=None):
    # Sets up the DEAP toolbox and sizes genomes dynamically for SISO/MISO/SIMO/MIMO
    ep_toolbox = base.Toolbox()
    model_type = hyperparameters.get("model_type", "MLP")

    n_u = 1 if (u_sample is None or u_sample.ndim == 1) else u_sample.shape[1]
    n_y = 1 if (y_sample is None or y_sample.ndim == 1) else y_sample.shape[1]

    custom_bounds = hyperparameters.get("search_bounds")
    layer_bounds = None

    if model_type == "ARMAX":
        total_parameters = get_armax_parameter_count(
            hyperparameters["na"], hyperparameters["nb"], hyperparameters.get("nc", 2), n_u=n_u, n_y=n_y
        )
        min_b, max_b = (float(custom_bounds[0]), float(custom_bounds[1])) if custom_bounds else (-0.5, 0.5)
        max_abs_b = max(abs(min_b), abs(max_b))
        layer_bounds = _build_armax_scaled_bounds(
            hyperparameters["na"], hyperparameters["nb"], hyperparameters.get("nc", 2),
            n_u=n_u, n_y=n_y, max_scale=max_abs_b
        )
    elif model_type == "SVR":
        total_parameters = get_svr_parameter_count(
            hyperparameters["na"], hyperparameters["nb"], n_u=n_u, n_y=n_y
        )
        min_b, max_b = 0.05, 2.0
    else:
        total_parameters = get_mlp_parameter_count(
            hyperparameters["na"], hyperparameters["nb"],
            hyperparameters.get("mlp_hidden_nodes", [5]), n_u=n_u, n_y=n_y
        )
        min_b, max_b = (float(custom_bounds[0]), float(custom_bounds[1])) if custom_bounds else (-0.5, 0.5)
        max_abs_b = max(abs(min_b), abs(max_b))
        layer_bounds = _build_mlp_xavier_bounds(
            hyperparameters["na"], hyperparameters["nb"],
            hyperparameters.get("mlp_hidden_nodes", [5]), n_u=n_u, n_y=n_y, max_scale=max_abs_b
        )

    pop_size = hyperparameters.get("pop_size", 150)
    worker_count = get_optimal_worker_count(hyperparameters, pop_size=pop_size)

    thread_pool = multiprocessing.Pool(
        processes=worker_count,
        initializer=_init_worker_process,
        initargs=(u_sample, y_sample, hyperparameters)
    )

    ep_toolbox.register("clone", fast_clone_genome)
    ep_toolbox.register(
        "individual", initialize_ep_genome, creator.EPIndividual,
        parameter_count=total_parameters, min_bound=min_b, max_bound=max_b,
        initial_sigma=hyperparameters.get("mutation_sigma", 0.5),
        layer_bounds=layer_bounds
    )
    ep_toolbox.register("population", tools.initRepeat, list, ep_toolbox.individual)
    cx_alpha = float(hyperparameters.get("crossover_alpha", 0.25))
    ep_toolbox.register("mate", apply_blend_crossover_with_strategy, alpha=cx_alpha)

    mut_type = hyperparameters.get("mutation_type", "Gaussian")
    ep_toolbox.register("mutate", apply_self_adaptive_mutation, mutation_type=mut_type)

    sel_type = str(hyperparameters.get("selection_type", "Truncation")).lower()
    if "tourn" in sel_type:
        t_size = max(2, int(hyperparameters.get("tournament_size", 4)))
        ep_toolbox.register("select", sel_ep_round_robin_tournament, tournsize=t_size)
    else:
        ep_toolbox.register("select", tools.selBest)

    return ep_toolbox, thread_pool


def _estimate_legacy_checkpoint_time(checkpoint_path):
    # Recovers true fold training duration for older checkpoints (where elapsed_time=0.0)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0.0
    try:
        ckpt_mtime = os.path.getmtime(checkpoint_path)
        folder = os.path.dirname(os.path.abspath(checkpoint_path))
        fname = os.path.basename(checkpoint_path)
        match = re.search(r"checkpoint_fold_(\d+)\.pkl", fname)
        if not match:
            return 0.0

        fold_idx = int(match.group(1))
        if fold_idx == 0:
            prev_file = os.path.join(folder, "run_metadata.json")
        else:
            prev_file = os.path.join(folder, f"checkpoint_fold_{fold_idx - 1}.pkl")

        if os.path.exists(prev_file):
            delta = ckpt_mtime - os.path.getmtime(prev_file)
            if 0.5 < delta < 21600.0:
                return float(delta)
    except Exception:
        pass
    return 0.0


def execute_evolutionary_programming(u_train, y_train, hyperparameters, stop_event=None, generation_callback=None):
    # Runs the main generational loop with cumulative elapsed-time tracking across resumes
    if stop_event and stop_event.is_set():
        return None, None

    model_type = hyperparameters.get("model_type", "MLP")
    n_u_svr = 1 if (u_train is None or u_train.ndim == 1) else u_train.shape[1]
    n_y_svr = 1 if (y_train is None or y_train.ndim == 1) else y_train.shape[1]
    na_cfg = int(hyperparameters.get("na", 0))
    nb_cfg = int(hyperparameters.get("nb", 0))
    nc_cfg = int(hyperparameters.get("nc", 2))
    stab_thresh = float(hyperparameters.get("stability_threshold", 0.98))
    total_lags_svr = max(1, na_cfg * n_y_svr + nb_cfg * n_u_svr)
    max_gamma_svr = 5.0 if n_u_svr == 1 else (4.0 / float(total_lags_svr))
    params_per_out_arm = na_cfg + (nb_cfg * n_u_svr) + nc_cfg

    def clamp_if_svr(ind):
        # Keep SVR parameters within valid bounds, and project ARMAX AR/MA polynomials
        # back onto the stable disc before evaluating fitness or saving checkpoints.
        if model_type == "SVR" and len(ind) >= 3:
            ind[0] = min(max(1e-3, abs(ind[0])), 50.0)
            ind[1] = min(max(1e-4, abs(ind[1])), 2.0)
            ind[2] = min(max(1e-4, abs(ind[2])), max_gamma_svr)
        elif model_type == "ARMAX" and len(ind) >= params_per_out_arm * n_y_svr:
            for out_ch in range(n_y_svr):
                offset = out_ch * params_per_out_arm
                if na_cfg > 0:
                    ar_s = ind[offset: offset + na_cfg]
                    if _max_root_magnitude_fast(ar_s, stab_thresh) > stab_thresh:
                        ind[offset: offset + na_cfg] = project_poly_to_stable_disc(ar_s, max_radius=stab_thresh).tolist()
                if nc_cfg > 0:
                    ma_off = offset + na_cfg + (nb_cfg * n_u_svr)
                    ma_s = ind[ma_off: ma_off + nc_cfg]
                    if _max_root_magnitude_fast(ma_s, stab_thresh) > stab_thresh:
                        ind[ma_off: ma_off + nc_cfg] = project_poly_to_stable_disc(ma_s, max_radius=stab_thresh).tolist()

    checkpoint_path = hyperparameters.get("checkpoint_file")
    restored_state = load_checkpoint_state(checkpoint_path, config=hyperparameters)

    session_start_time = time.perf_counter()
    prior_elapsed_time = 0.0

    # Fast path: If fold was already completed, return immediately without spawning a multiprocessing pool
    if restored_state:
        starting_generation = int(restored_state.get("generation", 0))
        prior_elapsed_time = float(restored_state.get("elapsed_time", 0.0))
        ep_logbook = restored_state.get("logbook")
        if prior_elapsed_time <= 0.0 and ep_logbook is not None:
            prior_elapsed_time = float(getattr(ep_logbook, "elapsed_time", 0.0))
        if prior_elapsed_time <= 0.0:
            prior_elapsed_time = _estimate_legacy_checkpoint_time(checkpoint_path)

        if restored_state.get("is_completed", False) or starting_generation >= hyperparameters["num_gen"]:
            print(f"[*] Fold already completed at generation {starting_generation} (Time: {prior_elapsed_time:.2f}s). Skipping.")
            if ep_logbook is not None:
                ep_logbook.elapsed_time = prior_elapsed_time
            hall_of_fame = restored_state.get("hall_of_fame")
            best_ind = list(hall_of_fame[0]) if hall_of_fame else []
            clamp_if_svr(best_ind)
            return best_ind, ep_logbook

    ep_toolbox, thread_pool = configure_ep_toolbox(hyperparameters, u_sample=u_train, y_sample=y_train)
    local_mlp_cache = None
    if model_type == "MLP" and u_train is not None and y_train is not None:
        local_mlp_cache = build_mlp_feature_matrix(
            u_train, y_train, int(hyperparameters["na"]), int(hyperparameters["nb"]), int(hyperparameters["nk"])
        )

    try:
        ep_statistics = tools.Statistics(lambda genome: genome.fitness.values)
        ep_statistics.register("avg", np.mean)
        ep_statistics.register("min", np.min)

        pop_size = max(2, int(hyperparameters.get("pop_size", 150)))
        num_workers = getattr(thread_pool, "_processes", get_optimal_worker_count(hyperparameters, pop_size=pop_size))
        chunk_sz = max(1, pop_size // (num_workers * 2))

        def current_total_elapsed():
            return prior_elapsed_time + (time.perf_counter() - session_start_time)

        if restored_state:
            population = restored_state["population"]
            starting_generation = restored_state["generation"]
            hall_of_fame = restored_state["hall_of_fame"]
            ep_logbook = restored_state["logbook"]
            session_start_time = time.perf_counter()
        else:
            population = ep_toolbox.population(n=pop_size)
            for ind in population:
                clamp_if_svr(ind)
            starting_generation = 0
            hall_of_fame = tools.HallOfFame(1)
            ep_logbook = tools.Logbook()
            ep_logbook.header = ["gen", "nevals"] + ep_statistics.fields

            if stop_event and stop_event.is_set():
                return None, ep_logbook

            fitness_scores = thread_pool.map(_evaluate_worker_genome, population, chunksize=chunk_sz)
            for genome, score in zip(population, fitness_scores):
                genome.fitness.values = score

            hall_of_fame.update(population)
            ep_logbook.record(gen=0, nevals=len(population), **ep_statistics.compile(population))
            best_mse_0 = float(ep_logbook.select('min')[-1])
            print(f"    -> Gen 0/{hyperparameters['num_gen']} | Best MSE: {best_mse_0:.6f}")
            if generation_callback:
                try:
                    generation_callback(0, hyperparameters['num_gen'], best_mse_0, current_total_elapsed())
                except Exception:
                    pass

            if checkpoint_path:
                total_el = current_total_elapsed()
                ep_logbook.elapsed_time = total_el
                save_checkpoint_state(
                    population, 0, hall_of_fame, ep_logbook, checkpoint_path,
                    config=hyperparameters, is_completed=False, elapsed_time=total_el
                )

        cxpb = float(hyperparameters.get("crossover_prob", 0.0))
        ckpt_freq = max(1, int(hyperparameters.get("checkpoint_freq", 10)))
        generation = starting_generation

        user_patience = hyperparameters.get("early_stop_patience")
        if user_patience is not None:
            user_p_int = int(user_patience)
            if user_p_int < 0:
                patience = None
            elif user_p_int == 0:
                patience = min(100, max(15, int(hyperparameters.get("num_gen", 100)) // 10))
            else:
                patience = user_p_int
        else:
            patience = min(100, max(15, int(hyperparameters.get("num_gen", 100)) // 10))

        user_tol = hyperparameters.get("early_stop_tol")
        tol = float(user_tol) if user_tol is not None else 1e-6

        for generation in range(starting_generation + 1, hyperparameters["num_gen"] + 1):
            if stop_event and stop_event.is_set():
                last_valid_gen = generation - 1
                total_el = current_total_elapsed()
                ep_logbook.elapsed_time = total_el
                print(f"[*] Optimization stopped at generation {last_valid_gen} (Elapsed: {total_el:.2f}s).")
                if checkpoint_path:
                    save_checkpoint_state(
                        population, last_valid_gen, hall_of_fame, ep_logbook, checkpoint_path,
                        config=hyperparameters, is_completed=False, elapsed_time=total_el
                    )
                break

            offspring_pool = [ep_toolbox.clone(genome) for genome in population]
            # Randomly shuffle candidates before mating
            random.shuffle(offspring_pool)

            for i in range(1, len(offspring_pool), 2):
                if random.random() < cxpb:
                    ep_toolbox.mate(offspring_pool[i - 1], offspring_pool[i])
                    del offspring_pool[i - 1].fitness.values
                    del offspring_pool[i].fitness.values

            for offspring in offspring_pool:
                ep_toolbox.mutate(offspring)
                clamp_if_svr(offspring)
                del offspring.fitness.values

            invalid_ind = [ind for ind in offspring_pool if not ind.fitness.valid]
            if invalid_ind:
                chunk_sz = max(1, len(invalid_ind) // (num_workers * 2))
                fitness_scores = thread_pool.map(_evaluate_worker_genome, invalid_ind, chunksize=chunk_sz)
                for genome, score in zip(invalid_ind, fitness_scores):
                    genome.fitness.values = score

            hall_of_fame.update(offspring_pool)
            is_final_gen = (generation == hyperparameters["num_gen"])

            # Decoupled Lamarckian Ridge refinement for MLP: refine incumbent elite + top candidate offspring
            refine_interval = max(1, int(hyperparameters.get("lamarckian_refine_freq", 5)))
            if model_type == "MLP" and (generation % refine_interval == 0 or is_final_gen):
                if len(hall_of_fame) > 0:
                    refined_elite = refine_best_genome_lamarckian(
                        hall_of_fame[0], u_train, y_train, hyperparameters, local_mlp_cache
                    )
                    if refined_elite.fitness.values[0] < hall_of_fame[0].fitness.values[0]:
                        hall_of_fame.update([refined_elite])
                        worst_off_idx = int(np.argmax([ind.fitness.values[0] for ind in offspring_pool]))
                        offspring_pool[worst_off_idx] = ep_toolbox.clone(refined_elite)

                top_k_refine = max(0, int(hyperparameters.get("refine_top_candidates", 6)))
                if top_k_refine > 0 and len(offspring_pool) > 0:
                    sorted_off_idx = np.argsort([ind.fitness.values[0] for ind in offspring_pool])
                    n_top = min(len(sorted_off_idx), max(1, top_k_refine // 2))
                    candidate_indices = list(sorted_off_idx[:n_top])
                    remaining_idx = list(sorted_off_idx[n_top:])
                    if remaining_idx and top_k_refine > n_top:
                        n_rand = min(len(remaining_idx), top_k_refine - n_top)
                        candidate_indices.extend(random.sample(remaining_idx, n_rand))

                    for cand_idx in candidate_indices:
                        refined_cand = refine_best_genome_lamarckian(
                            offspring_pool[cand_idx], u_train, y_train, hyperparameters, local_mlp_cache
                        )
                        if refined_cand.fitness.values[0] < offspring_pool[cand_idx].fitness.values[0]:
                            offspring_pool[cand_idx] = ep_toolbox.clone(refined_cand)
                            hall_of_fame.update([refined_cand])

            population[:] = ep_toolbox.select(population + offspring_pool, pop_size)
            # Elitism: make sure the best individual is preserved
            if hall_of_fame and len(hall_of_fame) > 0:
                if not any(ind == hall_of_fame[0] for ind in population):
                    worst_idx = int(np.argmax([ind.fitness.values[0] for ind in population]))
                    population[worst_idx] = ep_toolbox.clone(hall_of_fame[0])

            ep_logbook.record(gen=generation, nevals=len(invalid_ind), **ep_statistics.compile(population))
            best_mse_gen = float(ep_logbook.select('min')[-1])
            print(f"    -> Gen {generation}/{hyperparameters['num_gen']} | Best MSE: {best_mse_gen:.6f}")
            if generation_callback:
                try:
                    generation_callback(generation, hyperparameters['num_gen'], best_mse_gen, current_total_elapsed())
                except Exception:
                    pass

            if patience is not None:
                historical_mins = ep_logbook.select("min")
                if len(historical_mins) > patience and historical_mins[-1] < 1e5:
                    old_mse = float(historical_mins[-patience - 1])
                    cur_mse = float(historical_mins[-1])
                    abs_diff = abs(cur_mse - old_mse)
                    rel_tol = float(hyperparameters.get("early_stop_rel_tol", 1e-4))
                    rel_diff = abs_diff / max(1e-8, abs(old_mse))

                    if abs_diff < tol and rel_diff < rel_tol:
                        if model_type == "MLP" and len(hall_of_fame) > 0:
                            refined_final = refine_best_genome_lamarckian(
                                hall_of_fame[0], u_train, y_train, hyperparameters, local_mlp_cache
                            )
                            if refined_final.fitness.values[0] < hall_of_fame[0].fitness.values[0]:
                                hall_of_fame.update([refined_final])
                                worst_idx = int(np.argmax([ind.fitness.values[0] for ind in population]))
                                population[worst_idx] = ep_toolbox.clone(refined_final)
                                if len(ep_logbook) > 0:
                                    ep_logbook[-1]["min"] = float(refined_final.fitness.values[0])
                        total_el = current_total_elapsed()
                        ep_logbook.elapsed_time = total_el
                        print(f"[*] Converged early at generation {generation} (Patience: {patience}, Abs Tol: {tol:.1e}, Rel Tol: {rel_tol:.1e}, Elapsed: {total_el:.2f}s).")
                        if checkpoint_path:
                            save_checkpoint_state(
                                population, generation, hall_of_fame, ep_logbook, checkpoint_path,
                                config=hyperparameters, is_completed=True, elapsed_time=total_el
                            )
                        break

            if checkpoint_path and (generation % ckpt_freq == 0 or is_final_gen):
                total_el = current_total_elapsed()
                ep_logbook.elapsed_time = total_el
                save_checkpoint_state(
                    population, generation, hall_of_fame, ep_logbook, checkpoint_path,
                    config=hyperparameters, is_completed=is_final_gen, elapsed_time=total_el
                )

        ep_logbook.elapsed_time = current_total_elapsed()
        best_ind = list(hall_of_fame[0])
        clamp_if_svr(best_ind)
        return best_ind, ep_logbook

    finally:
        thread_pool.terminate()
        thread_pool.join()
