"""
optimization_benchmarks.py

Standard continuous test functions for evaluating
the evolutionary algorithm's convergence properties.
"""

import os
import time
import random
import multiprocessing
from datetime import datetime
import numpy as np
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
from diagnostic_plotter import generate_convergence_plot, generate_benchmark_boxplots


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


def execute_benchmark_trial(function_name, function_config, hyperparameters, dimensions=5, total_trials=10, output_dir="results_benchmarks", stop_event=None):
    """
    Executes independent optimization trials for a given benchmark topology.
    """
    min_bound, max_bound = function_config["search_bounds"]
    population_size = hyperparameters.get("pop_size", 150)
    max_generations = hyperparameters.get("num_gen", 100)
    initial_sigma = hyperparameters.get("mutation_sigma", 0.5)
    cxpb = hyperparameters.get("crossover_prob", 0.5)

    worker_count = get_optimal_worker_count(hyperparameters, pop_size=population_size)
    ep_toolbox = base.Toolbox()
    thread_pool = multiprocessing.Pool(processes=worker_count)
    ep_toolbox.register("clone", fast_clone_genome)
    ep_toolbox.register(
        "map",
        lambda func, iterable: thread_pool.map(
            func, iterable, chunksize=max(1, len(iterable) // (worker_count * 2))
        )
    )

    ep_toolbox.register(
        "individual",
        initialize_ep_genome,
        creator.EPIndividual,
        parameter_count=dimensions,
        min_bound=min_bound,
        max_bound=max_bound,
        initial_sigma=initial_sigma
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

    try:
        for trial_index in range(total_trials):
            if stop_event and stop_event.is_set():
                break

            enforce_reproducibility_seed(42 + trial_index)
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
                    # Enforce search boundaries so benchmark functions (like Schwefel) don't escape [-500, 500] and blow up
                    offspring[:] = np.clip(offspring, min_bound, max_bound).tolist()
                    del offspring.fitness.values

                invalid_ind = [ind for ind in offspring_pool if not ind.fitness.valid]
                fitness_scores = list(ep_toolbox.map(ep_toolbox.evaluate, invalid_ind))
                for genome, score in zip(invalid_ind, fitness_scores):
                    genome.fitness.values = score

                hall_of_fame.update(offspring_pool)
                population[:] = ep_toolbox.select(population + offspring_pool, population_size)
                # Elitism: make sure the best individual is preserved in the population
                if hall_of_fame and len(hall_of_fame) > 0:
                    if not any(ind == hall_of_fame[0] for ind in population):
                        worst_idx = int(np.argmax([ind.fitness.values[0] for ind in population]))
                        population[worst_idx] = ep_toolbox.clone(hall_of_fame[0])
                ep_logbook.record(gen=generation, nevals=len(invalid_ind), **ep_statistics.compile(population))

                # Stop early if best fitness hasn't improved over the last 20 generations
                historical_mins = ep_logbook.select("min")
                if len(historical_mins) > 20 and abs(historical_mins[-1] - historical_mins[-21]) < 1e-9:
                    break

            trial_elapsed_time = time.perf_counter() - trial_start_time
            optimal_fitness_scores.append(hall_of_fame[0].fitness.values[0])
            optimal_genomes.append(list(hall_of_fame[0]))
            trial_logbooks.append(ep_logbook)
            execution_times.append(trial_elapsed_time)

        if not optimal_fitness_scores:
            return None

        best_trial_index = int(np.argmin(optimal_fitness_scores))
        winning_trial_number = best_trial_index + 1

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
            "best_trial": winning_trial_number,
            "best_vector": optimal_genomes[best_trial_index],
            "all_scores": optimal_fitness_scores
        }

    finally:
        thread_pool.terminate()
        thread_pool.join()


def run_full_benchmark_suite(hyperparameters=None, dimensions=5, total_trials=10, output_dir=None, progress_callback=None, stop_event=None):
    """
    Orchestrates the benchmark suite, aggregating results across all test surfaces.
    """
    if hyperparameters is None:
        hyperparameters = load_hyperparameters("config.json")

    dimensions = int(hyperparameters.get("benchmark_dimensions", dimensions))
    total_trials = int(hyperparameters.get("benchmark_trials", total_trials))

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("results", f"results_benchmarks_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    report_lines = [
        "=== Optimization Benchmark Report ===",
        f"Dimensions (D): {dimensions} | Trials per Function: {total_trials}",
        f"Population Size: {hyperparameters.get('pop_size', 150)} | Max Generations: {hyperparameters.get('num_gen', 100)} | "
        f"Crossover Prob: {hyperparameters.get('crossover_prob', 0.5)} | Initial Sigma: {hyperparameters.get('mutation_sigma', 0.5)}\n"
    ]

    print("\n" + "=" * 70)
    print("Testing Algorithm on Standard Benchmark Functions")
    print("=" * 70)

    aggregated_scores_dict = {}

    for index, (function_name, function_config) in enumerate(TEST_FUNCTIONS.items(), start=1):
        if stop_event and stop_event.is_set():
            print("[*] Benchmark run stopped by user.")
            break

        if progress_callback:
            progress_callback(f"Running Benchmark {index}/{len(TEST_FUNCTIONS)}: {function_name}...")

        trial_results = execute_benchmark_trial(
            function_name, function_config, hyperparameters,
            dimensions=dimensions, total_trials=total_trials, output_dir=output_dir,
            stop_event=stop_event
        )
        if trial_results is None:
            break
        aggregated_scores_dict[function_name] = trial_results["all_scores"]

        console_line = (
            f"{function_name:<12} | Mean: {trial_results['mean_fitness']:>10.6f} +/- {trial_results['std_fitness']:<10.6f} | "
            f"Best: {trial_results['min_fitness']:>10.6f} (Trial {trial_results['best_trial']:02d}) | Avg Time: {trial_results['mean_time']:>5.2f}s"
        )
        print(console_line)

        report_lines.append(f"--- {function_name} ({function_config['description']}) ---")
        report_lines.append(f"Search Bounds        : {function_config['search_bounds']}")
        if function_name == "Rosenbrock":
            opt_loc = "[1, ..., 1]"
        elif function_name == "Schwefel":
            opt_loc = "[420.9687, ..., 420.9687]"
        else:
            opt_loc = "[0, ..., 0]"
        report_lines.append(f"Theoretical Optimum  : {function_config['theoretical_optimum']} at x* = {opt_loc}")
        report_lines.append(
            f"Mean Fitness +/- SD  : {trial_results['mean_fitness']:.6e} +/- {trial_results['std_fitness']:.6e}")
        report_lines.append(
            f"Best Fitness Found   : {trial_results['min_fitness']:.6e} (Achieved on Trial {trial_results['best_trial']} of {total_trials})")
        report_lines.append(f"Worst Fitness Found  : {trial_results['max_fitness']:.6e}")
        report_lines.append(f"Mean Execution Time  : {trial_results['mean_time']:.2f} seconds")

        vector_str = ", ".join([f"{v:.4f}" for v in trial_results["best_vector"]])
        report_lines.append(f"Best Solution Vector : [{vector_str}]\n")

    if aggregated_scores_dict:
        generate_benchmark_boxplots(aggregated_scores_dict, output_dir=output_dir)

    report_filepath = os.path.join(output_dir, "benchmark_report.txt")
    with open(report_filepath, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    print(f"\n[*] Tests complete. Report and plots saved to: {output_dir}")
    return report_filepath


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_full_benchmark_suite()