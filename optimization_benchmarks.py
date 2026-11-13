"""
optimization_benchmarks.py

Standard continuous test functions for evaluating
the evolutionary algorithm's convergence properties and comparing
against Differential Evolution (SciPy) under equal evaluation budgets.
"""

import os
import time
import random
import multiprocessing
from datetime import datetime
import numpy as np
import scipy.stats as stats
from scipy.optimize import differential_evolution
from deap import base, creator, tools, benchmarks

from ep_algorithm_core import (
    initialize_ep_genome,
    apply_self_adaptive_mutation,
    apply_self_adaptive_gaussian_mutation,
    apply_blend_crossover_with_strategy,
    fast_clone_genome,
    sel_ep_round_robin_tournament,
    get_optimal_worker_count
)
from config_handler import load_hyperparameters
from diagnostic_plotter import (
    generate_convergence_plot,
    generate_benchmark_boxplots,
    generate_benchmark_convergence_comparison
)


def enforce_reproducibility_seed(seed_value=42):
    # Set seeds for numpy and random so runs can be reproduced
    np.random.seed(seed_value)
    random.seed(seed_value)


TEST_FUNCTIONS = {
    "Sphere": {
        "objective_function": benchmarks.sphere,
        "search_bounds": (-5.12, 5.12),
        "theoretical_optimum": 0.0,
        "description": "Convex quadratic unimodal function"
    },
    "Rosenbrock": {
        "objective_function": benchmarks.rosenbrock,
        "search_bounds": (-2.048, 2.048),
        "theoretical_optimum": 0.0,
        "description": "Non-convex banana valley testing variable dependency"
    },
    "Rastrigin": {
        "objective_function": benchmarks.rastrigin,
        "search_bounds": (-5.12, 5.12),
        "theoretical_optimum": 0.0,
        "description": "Highly multimodal function with deep regular local minima"
    },
    "Ackley": {
        "objective_function": benchmarks.ackley,
        "search_bounds": (-5.0, 5.0),
        "theoretical_optimum": 0.0,
        "description": "Nearly flat outer region with steep central basin"
    },
    "Griewank": {
        "objective_function": benchmarks.griewank,
        "search_bounds": (-600.0, 600.0),
        "theoretical_optimum": 0.0,
        "description": "Multimodal function testing component-coupling via product of cosines"
    },
    "Schwefel": {
        "objective_function": benchmarks.schwefel,
        "search_bounds": (-500.0, 500.0),
        "theoretical_optimum": 0.0,
        "description": "Deceptive multimodal function with distant global optimum"
    }
}

BENCHMARK_PRECISION_THRESHOLDS = {
    "Sphere": 1e-4,
    "Rosenbrock": 1.0,
    "Rastrigin": 1.0,
    "Ackley": 1e-2,
    "Griewank": 0.1,
    "Schwefel": 100.0
}


def compute_vargha_delaney_a12(sample1, sample2):
    """
    Computes the Vargha-Delaney A12 non-parametric effect size for minimization problems.
    A12 > 0.5 indicates sample1 (EP) achieves smaller (better) values than sample2 (DE).
    """
    m, n = len(sample1), len(sample2)
    if m == 0 or n == 0:
        return 0.5, "Negligible"
    count = 0.0
    for x in sample1:
        for y in sample2:
            if x < y:
                count += 1.0
            elif x == y:
                count += 0.5
    a12 = count / (m * n)
    diff = abs(a12 - 0.5)
    if diff >= 0.21:
        magnitude = "Large"
    elif diff >= 0.14:
        magnitude = "Medium"
    elif diff >= 0.06:
        magnitude = "Small"
    else:
        magnitude = "Negligible"
    return a12, magnitude


def EvaluateBenchmarkComparison(ep_scores: np.ndarray, de_scores: np.ndarray, tol: float = 1e-9) -> dict:
    """
    Computes statistical metrics (Wilcoxon / Mann-Whitney, Vargha-Delaney A12)
    with strict handling for numerical ties near zero.
    """
    ep_scores = np.asarray(ep_scores, dtype=float)
    de_scores = np.asarray(de_scores, dtype=float)
    ep_mean = float(np.mean(ep_scores))
    de_mean = float(np.mean(de_scores))
    diff = abs(ep_mean - de_mean)

    # Standard statistics calculation
    if np.array_equal(ep_scores, de_scores) or np.allclose(ep_scores, de_scores, atol=tol):
        p_val = 1.0
        u_stat = 0.0
        a12 = 0.50
        effect_mag = "Negligible"
        sig_marker = "ns"
        winner = "Tie"
    else:
        # Perform Wilcoxon rank-sum / Mann-Whitney U test
        try:
            stat, p_val = stats.mannwhitneyu(ep_scores, de_scores, alternative="two-sided")
            u_stat = float(stat)
            p_val = float(p_val)
        except (ValueError, Exception):
            p_val = 1.0
            u_stat = 0.0

        a12, effect_mag = compute_vargha_delaney_a12(ep_scores, de_scores)

        if diff <= tol:
            winner = "Tie"
            p_val = 1.0
            a12 = 0.50
            effect_mag = "Negligible"
            sig_marker = "ns"
        elif ep_mean < de_mean:
            winner = "EP"
            sig_marker = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        else:
            winner = "DE"
            sig_marker = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))

    return {
        "ep_mean": ep_mean,
        "de_mean": de_mean,
        "winner": winner,
        "p_val": p_val,
        "u_stat": u_stat,
        "a12": a12,
        "effect_mag": effect_mag,
        "sig_marker": sig_marker
    }


evaluate_benchmark_comparison = EvaluateBenchmarkComparison


def execute_benchmark_trial(
    function_name,
    function_config,
    hyperparameters,
    dimensions=5,
    total_trials=10,
    output_dir="results_benchmarks",
    stop_event=None,
    allow_early_stop=False
):
    """
    Executes independent Evolutionary Programming (EP) trials for a given benchmark surface.
    By default allow_early_stop=False so every trial executes the exact same FE budget
    and produces full-length convergence curves for fair comparison against DE.
    Uses direct in-process evaluation by default on fast analytical functions to avoid
    inter-process serialization overhead against single-process SciPy DE.
    """
    min_bound, max_bound = function_config["search_bounds"]
    population_size = int(hyperparameters.get("pop_size", 150))
    max_generations = int(hyperparameters.get("num_gen", 100))
    initial_sigma = float(hyperparameters.get("mutation_sigma", 0.5))
    cxpb = float(hyperparameters.get("crossover_prob", 0.5))

    domain_span = float(max_bound - min_bound)
    scaled_sigma = initial_sigma * max(1.0, domain_span / 10.0)

    use_mp = bool(hyperparameters.get("benchmark_multiprocessing", False))
    thread_pool = None
    ep_toolbox = base.Toolbox()
    ep_toolbox.register("clone", fast_clone_genome)

    if use_mp:
        worker_count = get_optimal_worker_count(hyperparameters, pop_size=population_size)
        thread_pool = multiprocessing.Pool(processes=worker_count)
        ep_toolbox.register(
            "map",
            lambda func, iterable: thread_pool.map(
                func, iterable, chunksize=max(1, len(iterable) // (worker_count * 2))
            )
        )
    else:
        ep_toolbox.register("map", map)

    ep_toolbox.register(
        "individual",
        initialize_ep_genome,
        creator.EPIndividual,
        parameter_count=dimensions,
        min_bound=min_bound,
        max_bound=max_bound,
        initial_sigma=scaled_sigma
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

    ep_toolbox.register("evaluate", function_config["objective_function"])

    ep_statistics = tools.Statistics(lambda genome: genome.fitness.values)
    ep_statistics.register("avg", np.mean)
    ep_statistics.register("min", np.min)

    optimal_fitness_scores = []
    optimal_genomes = []
    trial_logbooks = []
    execution_times = []
    total_fevals_list = []
    convergence_matrix = []

    try:
        base_seed = int(hyperparameters.get("random_seed", 42))
        for trial_index in range(total_trials):
            if stop_event and stop_event.is_set():
                break

            enforce_reproducibility_seed(base_seed + trial_index)
            trial_start_time = time.perf_counter()

            population = ep_toolbox.population(n=population_size)
            hall_of_fame = tools.HallOfFame(1)
            ep_logbook = tools.Logbook()
            ep_logbook.header = ["gen", "nevals"] + ep_statistics.fields

            fitness_scores = list(ep_toolbox.map(ep_toolbox.evaluate, population))
            for genome, score in zip(population, fitness_scores):
                genome.fitness.values = score

            hall_of_fame.update(population)
            ep_logbook.record(gen=0, nevals=len(population), **ep_statistics.compile(population))

            for generation in range(1, max_generations + 1):
                if stop_event and stop_event.is_set():
                    break

                offspring_pool = [ep_toolbox.clone(genome) for genome in population]
                random.shuffle(offspring_pool)

                for i in range(1, len(offspring_pool), 2):
                    if random.random() < cxpb:
                        ep_toolbox.mate(offspring_pool[i - 1], offspring_pool[i])
                        del offspring_pool[i - 1].fitness.values
                        del offspring_pool[i].fitness.values

                for offspring in offspring_pool:
                    ep_toolbox.mutate(offspring)
                    offspring[:] = np.clip(offspring, min_bound, max_bound).tolist()
                    del offspring.fitness.values

                invalid_ind = [ind for ind in offspring_pool if not ind.fitness.valid]
                fitness_scores = list(ep_toolbox.map(ep_toolbox.evaluate, invalid_ind))
                for genome, score in zip(invalid_ind, fitness_scores):
                    genome.fitness.values = score

                hall_of_fame.update(offspring_pool)
                population[:] = ep_toolbox.select(population + offspring_pool, population_size)
                if hall_of_fame and len(hall_of_fame) > 0:
                    if not any(ind == hall_of_fame[0] for ind in population):
                        worst_idx = int(np.argmax([ind.fitness.values[0] for ind in population]))
                        population[worst_idx] = ep_toolbox.clone(hall_of_fame[0])
                ep_logbook.record(gen=generation, nevals=len(invalid_ind), **ep_statistics.compile(population))

                if allow_early_stop:
                    historical_mins = ep_logbook.select("min")
                    if len(historical_mins) > 20 and abs(historical_mins[-1] - historical_mins[-21]) < 1e-12:
                        break

            trial_elapsed_time = time.perf_counter() - trial_start_time
            optimal_fitness_scores.append(float(hall_of_fame[0].fitness.values[0]))
            optimal_genomes.append(list(hall_of_fame[0]))
            trial_logbooks.append(ep_logbook)
            execution_times.append(trial_elapsed_time)
            total_fevals_list.append(int(np.sum(ep_logbook.select("nevals"))))

            mins_seq = [float(v) for v in ep_logbook.select("min")]
            target_len = max_generations + 1
            if len(mins_seq) < target_len:
                mins_seq.extend([mins_seq[-1]] * (target_len - len(mins_seq)))
            else:
                mins_seq = mins_seq[:target_len]
            convergence_matrix.append(mins_seq)

        if not optimal_fitness_scores:
            return None

        best_trial_index = int(np.argmin(optimal_fitness_scores))
        winning_trial_number = best_trial_index + 1

        if output_dir:
            generate_convergence_plot(
                trial_logbooks[best_trial_index],
                f"Benchmark_{function_name}",
                run_index=winning_trial_number,
                output_dir=output_dir,
                run_label="Trial",
                y_axis_label="Objective Value",
                best_curve_label="Best Fitness",
                title_suffix=f" (Best of {total_trials})"
            )

        return {
            "mean_fitness": float(np.mean(optimal_fitness_scores)),
            "std_fitness": float(np.std(optimal_fitness_scores)),
            "min_fitness": float(np.min(optimal_fitness_scores)),
            "max_fitness": float(np.max(optimal_fitness_scores)),
            "mean_time": float(np.mean(execution_times)),
            "std_time": float(np.std(execution_times)),
            "mean_fevals": int(np.mean(total_fevals_list)),
            "best_trial": winning_trial_number,
            "best_vector": optimal_genomes[best_trial_index],
            "all_scores": optimal_fitness_scores,
            "convergence_matrix": np.asarray(convergence_matrix, dtype=float)
        }

    finally:
        if thread_pool is not None:
            thread_pool.terminate()
            thread_pool.join()


def execute_de_benchmark_trial(function_name, function_config, hyperparameters, dimensions=5, total_trials=10, stop_event=None):
    """
    Executes SciPy Differential Evolution (best1bin) on the exact same benchmark function,
    search bounds, and dimensions with a matched number of fitness evaluations (FE).
    Adjusts maxiter when population_size is not an integer multiple of dimensions (e.g. D=20)
    so total FE matches EP's population_size * (max_generations + 1).
    """
    min_bound, max_bound = function_config["search_bounds"]
    population_size = int(hyperparameters.get("pop_size", 150))
    max_generations = int(hyperparameters.get("num_gen", 100))
    base_seed = int(hyperparameters.get("random_seed", 42))

    de_popsize_mult = max(1, population_size // max(1, dimensions))
    actual_de_pop = de_popsize_mult * max(1, dimensions)
    de_maxiter = max(1, int(round((population_size * max_generations) / float(actual_de_pop))))

    bounds_list = [(min_bound, max_bound)] * dimensions
    deap_obj = function_config["objective_function"]

    def scalar_objective(x):
        return float(deap_obj(x)[0])

    optimal_fitness_scores = []
    optimal_genomes = []
    execution_times = []
    total_fevals_list = []
    convergence_matrix = []

    for trial_index in range(total_trials):
        if stop_event and stop_event.is_set():
            break

        trial_seed = base_seed + trial_index
        enforce_reproducibility_seed(trial_seed)

        init_pop = np.random.uniform(min_bound, max_bound, size=(actual_de_pop, dimensions))
        gen0_best = float(np.min([scalar_objective(ind) for ind in init_pop]))
        history = [gen0_best]

        def de_callback(xk, convergence=None):
            if stop_event and stop_event.is_set():
                return True
            val = scalar_objective(xk)
            best_so_far = min(history[-1], val) if history else val
            history.append(best_so_far)
            return False

        t0 = time.perf_counter()
        res = differential_evolution(
            scalar_objective,
            bounds=bounds_list,
            strategy="best1bin",
            maxiter=de_maxiter,
            popsize=de_popsize_mult,
            init=init_pop,
            tol=0.0,
            atol=0.0,
            polish=False,
            seed=trial_seed,
            callback=de_callback
        )
        elapsed = time.perf_counter() - t0

        final_best = min(float(res.fun), history[-1] if history else float(res.fun))
        target_len = max_generations + 1
        if len(history) != target_len:
            x_old = np.linspace(0.0, 1.0, len(history))
            x_new = np.linspace(0.0, 1.0, target_len)
            history = np.interp(x_new, x_old, np.asarray(history, dtype=float)).tolist()
        history[-1] = final_best

        optimal_fitness_scores.append(final_best)
        optimal_genomes.append(res.x.tolist())
        execution_times.append(elapsed)
        total_fevals_list.append(int(res.nfev))
        convergence_matrix.append(history)

    if not optimal_fitness_scores:
        return None

    best_trial_index = int(np.argmin(optimal_fitness_scores))
    return {
        "mean_fitness": float(np.mean(optimal_fitness_scores)),
        "std_fitness": float(np.std(optimal_fitness_scores)),
        "min_fitness": float(np.min(optimal_fitness_scores)),
        "max_fitness": float(np.max(optimal_fitness_scores)),
        "mean_time": float(np.mean(execution_times)),
        "std_time": float(np.std(execution_times)),
        "mean_fevals": int(np.mean(total_fevals_list)),
        "actual_de_pop": actual_de_pop,
        "best_trial": best_trial_index + 1,
        "best_vector": optimal_genomes[best_trial_index],
        "all_scores": optimal_fitness_scores,
        "convergence_matrix": np.asarray(convergence_matrix, dtype=float)
    }


def _export_benchmark_tables(summary_rows, output_dir, dimensions, total_trials, hyperparameters):
    csv_path = os.path.join(output_dir, "benchmark_comparison_table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Function,Algorithm,Dimensions,Trials,Fitness_Evaluations,Mean_Fitness,Std_Fitness,Best_Fitness,Worst_Fitness,Mean_Time_Sec,Std_Time_Sec,P_Value,Significance,A12,Effect,Target_Eps,Success_Rate_Pct,Mean_Hitting_FE\n")
        for r in summary_rows:
            f.write(
                f"{r['Function']},{r['Algorithm']},{dimensions},{total_trials},{r['FE']},"
                f"{r['Mean']:.6e},{r['Std']:.6e},{r['Best']:.6e},{r['Worst']:.6e},"
                f"{r['Time_Mean']:.4f},{r['Time_Std']:.4f},{r.get('P_Value', 1.0):.4e},"
                f"{r.get('Significance', 'ns')},{r.get('A12', 0.5):.4f},{r.get('Effect', 'Negligible')},"
                f"{r.get('Target_Eps', 1e-2):.4e},{r.get('Success_Rate_Pct', 0.0):.1f},"
                f"{r.get('Mean_Hitting_FE', float('nan')):.1f}\n"
            )

    tex_path = os.path.join(output_dir, "benchmark_comparison_table.tex")
    lines = [
        r"% Auto-generated Benchmark Comparison Table (EP vs. Differential Evolution)",
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Benchmark Functions Comparison ($D={dimensions}$, {total_trials} Trials): Proposed EP vs. Differential Evolution (DE)}}",
        r"\label{tab:benchmark_ep_vs_de}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Function} & \textbf{Algorithm} & \textbf{Mean} & \textbf{Std (SD)} & \textbf{Best} & \textbf{FE} & \textbf{Time (s)} \\",
        r"\midrule"
    ]
    for idx, r in enumerate(summary_rows):
        lines.append(
            rf"{r['Function']} & {r['Algorithm']} & {r['Mean']:.3e} & {r['Std']:.3e} & {r['Best']:.3e} & {r['FE']} & {r['Time_Mean']:.2f} \\"
        )
        if idx % 2 == 1 and idx < len(summary_rows) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        ""
    ])
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Export dedicated statistical significance table
    stat_tex_path = os.path.join(output_dir, "benchmark_statistical_significance.tex")
    stat_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Non-Parametric Statistical Comparison of EP vs. DE ($D={dimensions}$, $N={total_trials}$ Trials)}}",
        r"\label{tab:benchmark_statistical_significance}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Function} & \textbf{EP ($\mu \pm \sigma$)} & \textbf{DE ($\mu \pm \sigma$)} & \textbf{$p$-value} & \textbf{Signif.} & \textbf{Effect ($A_{12}$)} & \textbf{Winner} \\",
        r"\midrule"
    ]
    for i in range(0, len(summary_rows), 2):
        ep_r = summary_rows[i]
        de_r = summary_rows[i + 1]
        fn = ep_r["Function"]
        pv = ep_r.get("P_Value", 1.0)
        pv_tex = "1.0000" if abs(pv - 1.0) < 1e-6 else (f"{pv:.2e}" if pv >= 1e-4 else "< 10^{-4}")
        stat_lines.append(
            f"{fn} & {ep_r['Mean']:.3e} $\\pm$ {ep_r['Std']:.2e} & "
            f"{de_r['Mean']:.3e} $\\pm$ {de_r['Std']:.2e} & "
            f"${pv_tex}$ & {ep_r.get('Significance', 'ns')} & {ep_r.get('A12', 0.5):.3f} ({ep_r.get('Effect', 'Negligible')}) & {ep_r.get('Winner', 'Tie')} \\\\"
        )
    stat_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        ""
    ])
    with open(stat_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stat_lines))

    # Export dedicated success rates table
    sr_tex_path = os.path.join(output_dir, "benchmark_success_rates.tex")
    sr_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Empirical Success Rates and Mean Hitting Times to Target Precision $\epsilon$ ($D={dimensions}$)}}",
        r"\label{tab:benchmark_success_rates}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Function} & \textbf{Threshold ($\epsilon$)} & \textbf{EP SR (\%)} & \textbf{DE SR (\%)} & \textbf{EP Mean $FE_{\text{hit}}$} & \textbf{DE Mean $FE_{\text{hit}}$} \\",
        r"\midrule"
    ]
    for i in range(0, len(summary_rows), 2):
        ep_r = summary_rows[i]
        de_r = summary_rows[i + 1]
        fn = ep_r["Function"]
        t_eps = ep_r.get("Target_Eps", 1e-2)
        ep_fe = ep_r.get("Mean_Hitting_FE", float("nan"))
        de_fe = de_r.get("Mean_Hitting_FE", float("nan"))
        ep_fe_str = f"{ep_fe:.0f}" if not np.isnan(ep_fe) else "--"
        de_fe_str = f"{de_fe:.0f}" if not np.isnan(de_fe) else "--"
        sr_lines.append(
            f"{fn} & {t_eps:.1e} & "
            f"{ep_r.get('Success_Rate_Pct', 0.0):.1f}\\% & {de_r.get('Success_Rate_Pct', 0.0):.1f}\\% & "
            f"{ep_fe_str} & {de_fe_str} \\\\"
        )
    sr_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        ""
    ])
    with open(sr_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sr_lines))


def run_full_benchmark_suite(hyperparameters=None, dimensions=5, total_trials=10, output_dir=None, progress_callback=None, stop_event=None):
    """
    Orchestrates the benchmark suite, running both the proposed Evolutionary Programming (EP)
    algorithm and SciPy Differential Evolution (DE) under matched computational budgets.
    """
    if hyperparameters is None:
        hyperparameters = load_hyperparameters("config.json")

    dimensions = int(hyperparameters.get("benchmark_dimensions", dimensions))
    total_trials = int(hyperparameters.get("benchmark_trials", total_trials))

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("results", f"results_benchmarks_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    mut_type = hyperparameters.get("mutation_type", "Gaussian")
    sel_type = hyperparameters.get("selection_type", "Truncation")
    ep_label = f"EP ({mut_type}/{sel_type})"

    report_lines = [
        "=== Evolutionary Programming (EP) vs. Differential Evolution (DE) Benchmark Report ===",
        f"Dimensions (D): {dimensions} | Independent Trials per Topology: {total_trials}",
        f"Population Size: {hyperparameters.get('pop_size', 150)} | Max Generations: {hyperparameters.get('num_gen', 100)} | "
        f"Mutation: {mut_type} (sigma={hyperparameters.get('mutation_sigma', 0.5)}) | Selection: {sel_type} | "
        f"Crossover Prob: {hyperparameters.get('crossover_prob', 0.5)}\n"
    ]

    print("\n" + "=" * 86)
    print(f"Testing {ep_label} vs. Differential Evolution (DE) on Standard Benchmark Functions")
    print("=" * 86)

    aggregated_scores_dict = {}
    convergence_histories_dict = {}
    summary_rows = []

    for index, (function_name, function_config) in enumerate(TEST_FUNCTIONS.items(), start=1):
        if stop_event and stop_event.is_set():
            print("[*] Benchmark run stopped by user.")
            break

        if progress_callback:
            progress_callback(f"Running Benchmark {index}/{len(TEST_FUNCTIONS)}: {function_name} (EP & DE)...")

        ep_results = execute_benchmark_trial(
            function_name, function_config, hyperparameters,
            dimensions=dimensions, total_trials=total_trials, output_dir=output_dir,
            stop_event=stop_event, allow_early_stop=False
        )
        if ep_results is None:
            break

        de_results = execute_de_benchmark_trial(
            function_name, function_config, hyperparameters,
            dimensions=dimensions, total_trials=total_trials,
            stop_event=stop_event
        )
        if de_results is None:
            break

        ep_scores = ep_results["all_scores"]
        de_scores = de_results["all_scores"]

        # 1. Non-parametric hypothesis testing & effect size with strict numerical tie reconciliation
        comp_metrics = EvaluateBenchmarkComparison(ep_scores, de_scores, tol=1e-9)
        p_val = comp_metrics["p_val"]
        u_stat = comp_metrics["u_stat"]
        sig_marker = comp_metrics["sig_marker"]
        a12 = comp_metrics["a12"]
        effect_mag = comp_metrics["effect_mag"]
        winner = comp_metrics["winner"]

        # 2. Precision Threshold & Success Rate (Generation 0 evaluates initial pop, so +1)
        target_eps = BENCHMARK_PRECISION_THRESHOLDS.get(function_name, 1e-2)
        pop_sz = int(hyperparameters.get("pop_size", 150))
        de_pop_sz = int(de_results.get("actual_de_pop", pop_sz))
        ep_hits = [
            int((np.where(trial_hist <= target_eps)[0][0] + 1) * pop_sz)
            for trial_hist in ep_results["convergence_matrix"]
            if len(np.where(trial_hist <= target_eps)[0]) > 0
        ]
        de_hits = [
            int((np.where(trial_hist <= target_eps)[0][0] + 1) * de_pop_sz)
            for trial_hist in de_results["convergence_matrix"]
            if len(np.where(trial_hist <= target_eps)[0]) > 0
        ]
        ep_sr = (len(ep_hits) / total_trials) * 100.0
        de_sr = (len(de_hits) / total_trials) * 100.0
        ep_hit_fe = float(np.mean(ep_hits)) if ep_hits else float("nan")
        de_hit_fe = float(np.mean(de_hits)) if de_hits else float("nan")

        aggregated_scores_dict[function_name] = {
            ep_label: ep_scores,
            "Differential Evolution (DE)": de_scores
        }
        convergence_histories_dict[function_name] = {
            ep_label: ep_results["convergence_matrix"],
            "Differential Evolution (DE)": de_results["convergence_matrix"]
        }

        summary_rows.append({
            "Function": function_name,
            "Algorithm": ep_label,
            "FE": ep_results["mean_fevals"],
            "Mean": ep_results["mean_fitness"],
            "Std": ep_results["std_fitness"],
            "Best": ep_results["min_fitness"],
            "Worst": ep_results["max_fitness"],
            "Time_Mean": ep_results["mean_time"],
            "Time_Std": ep_results["std_time"],
            "P_Value": p_val,
            "Significance": sig_marker,
            "A12": a12,
            "Effect": effect_mag,
            "Winner": winner,
            "Target_Eps": target_eps,
            "Success_Rate_Pct": ep_sr,
            "Mean_Hitting_FE": ep_hit_fe
        })
        summary_rows.append({
            "Function": function_name,
            "Algorithm": "Differential Evolution",
            "FE": de_results["mean_fevals"],
            "Mean": de_results["mean_fitness"],
            "Std": de_results["std_fitness"],
            "Best": de_results["min_fitness"],
            "Worst": de_results["max_fitness"],
            "Time_Mean": de_results["mean_time"],
            "Time_Std": de_results["std_time"],
            "P_Value": p_val,
            "Significance": sig_marker,
            "A12": a12,
            "Effect": effect_mag,
            "Winner": winner,
            "Target_Eps": target_eps,
            "Success_Rate_Pct": de_sr,
            "Mean_Hitting_FE": de_hit_fe
        })

        print(
            f"{function_name:<11} | [EP] Mean: {ep_results['mean_fitness']:>10.4e} +/- {ep_results['std_fitness']:<10.4e} | "
            f"Best: {ep_results['min_fitness']:>10.4e} | FE: {ep_results['mean_fevals']:>6d} | Time: {ep_results['mean_time']:>5.2f}s"
        )
        print(
            f"{'':<11} | [DE] Mean: {de_results['mean_fitness']:>10.4e} +/- {de_results['std_fitness']:<10.4e} | "
            f"Best: {de_results['min_fitness']:>10.4e} | FE: {de_results['mean_fevals']:>6d} | Time: {de_results['mean_time']:>5.2f}s"
        )
        print("-" * 86)

        if function_name == "Rosenbrock":
            opt_loc = "[1, ..., 1]"
        elif function_name == "Schwefel":
            opt_loc = "[420.9687, ..., 420.9687]"
        else:
            opt_loc = "[0, ..., 0]"

        report_lines.append(f"--- {function_name} Topology ({function_config['description']}) ---")
        report_lines.append(f"Search Bounds        : {function_config['search_bounds']}")
        report_lines.append(f"Theoretical Optimum  : {function_config['theoretical_optimum']} at x* = {opt_loc}")
        report_lines.append(f"Target Precision (ε) : {target_eps}")

        ep_vec_str = ", ".join([f"{v:.4f}" for v in ep_results["best_vector"]])
        report_lines.append(f"\n  [{ep_label}]")
        report_lines.append(f"    Mean Fitness +/- SD  : {ep_results['mean_fitness']:.6e} +/- {ep_results['std_fitness']:.6e}")
        report_lines.append(f"    Best Fitness Found   : {ep_results['min_fitness']:.6e} (Achieved on Trial {ep_results['best_trial']} of {total_trials})")
        report_lines.append(f"    Success Rate (<= ε)  : {ep_sr:.1f}% ({len(ep_hits)}/{total_trials} trials)")
        report_lines.append(f"    Mean Hitting FE      : {ep_hit_fe if not np.isnan(ep_hit_fe) else 'Not Reached'}")
        report_lines.append(f"    Worst Fitness Found  : {ep_results['max_fitness']:.6e}")
        report_lines.append(f"    Fitness Evaluations  : {ep_results['mean_fevals']}")
        report_lines.append(f"    Mean Execution Time  : {ep_results['mean_time']:.2f} seconds")
        report_lines.append(f"    Best Solution Vector : [{ep_vec_str}]")

        de_vec_str = ", ".join([f"{v:.4f}" for v in de_results["best_vector"]])
        report_lines.append(f"\n  [Differential Evolution (DE)]")
        report_lines.append(f"    Mean Fitness +/- SD  : {de_results['mean_fitness']:.6e} +/- {de_results['std_fitness']:.6e}")
        report_lines.append(f"    Best Fitness Found   : {de_results['min_fitness']:.6e} (Achieved on Trial {de_results['best_trial']} of {total_trials})")
        report_lines.append(f"    Success Rate (<= ε)  : {de_sr:.1f}% ({len(de_hits)}/{total_trials} trials)")
        report_lines.append(f"    Mean Hitting FE      : {de_hit_fe if not np.isnan(de_hit_fe) else 'Not Reached'}")
        report_lines.append(f"    Worst Fitness Found  : {de_results['max_fitness']:.6e}")
        report_lines.append(f"    Fitness Evaluations  : {de_results['mean_fevals']}")
        report_lines.append(f"    Mean Execution Time  : {de_results['mean_time']:.2f} seconds")
        report_lines.append(f"    Best Solution Vector : [{de_vec_str}]")

        report_lines.append(f"\n  [Statistical Significance & Effect Size]")
        report_lines.append(f"    Mann-Whitney U Test  : U = {u_stat:.1f}, p = {p_val:.4e} ({sig_marker})")
        report_lines.append(f"    Vargha-Delaney A12   : {a12:.4f} ({effect_mag} effect)")
        report_lines.append(f"    Topology Winner      : {winner}\n")

    if summary_rows:
        table_lines = [
            "=================================================================================================================================================",
            "                                  HEAD-TO-HEAD STATISTICAL SUMMARY COMPARISON TABLE (EP vs. Differential Evolution)                              ",
            "=================================================================================================================================================",
            f"{'Function':<11} | {'EP Mean +/- SD':<23} | {'DE Mean +/- SD':<23} | {'p-value':<10} | {'Sig':<4} | {'A12':<6} | {'Effect':<10} | {'FE':>6} | {'Winner'}",
            "-" * 145
        ]
        for i in range(0, len(summary_rows), 2):
            ep_r = summary_rows[i]
            de_r = summary_rows[i + 1]
            fn = ep_r["Function"]
            ep_ms = f"{ep_r['Mean']:.3e} +/- {ep_r['Std']:.2e}"
            de_ms = f"{de_r['Mean']:.3e} +/- {de_r['Std']:.2e}"
            pv = ep_r.get("P_Value", 1.0)
            pv_str = f"{pv:.2e}" if pv >= 1e-4 else "< 1e-4"
            table_lines.append(
                f"{fn:<11} | {ep_ms:<23} | {de_ms:<23} | {pv_str:<10} | {ep_r.get('Significance', 'ns'):<4} | "
                f"{ep_r.get('A12', 0.5):<6.3f} | {ep_r.get('Effect', 'Negligible'):<10} | {ep_r['FE']:>6d} | {ep_r.get('Winner', 'Tie')}"
            )
        table_lines.append("=" * 145 + "\n")
        print("\n" + "\n".join(table_lines))
        report_lines.extend(table_lines)

    if aggregated_scores_dict:
        generate_benchmark_boxplots(aggregated_scores_dict, output_dir=output_dir)
    if convergence_histories_dict:
        generate_benchmark_convergence_comparison(convergence_histories_dict, output_dir=output_dir)
    if summary_rows:
        _export_benchmark_tables(summary_rows, output_dir, dimensions, total_trials, hyperparameters)

    report_filepath = os.path.join(output_dir, "benchmark_report.txt")
    with open(report_filepath, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    print(f"\n[*] Benchmarks complete. Tables, convergence plots, and boxplots saved to: {output_dir}")
    return report_filepath


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_full_benchmark_suite()
