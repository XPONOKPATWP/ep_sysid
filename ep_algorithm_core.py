"""
ep_algorithm_core.py

Optimization engine using the DEAP framework to minimize Mean Squared Error.
"""

import numpy as np
import random
import multiprocessing
from deap import base, creator, tools

from armax_simulator import simulate_armax_system
from validation_metrics import compute_mse
from config_handler import save_checkpoint_state, load_checkpoint_state

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if not hasattr(creator, "EPIndividual"):
    creator.create("EPIndividual", list, fitness=creator.FitnessMin, strategy=None)


def initialize_ep_genome(genome_class, parameter_count, min_bound=-2.0, max_bound=2.0, initial_sigma=0.5):
    """
    Instantiates a candidate parameter vector (genome) and its associated strategy vector.
    """
    genome = genome_class(np.random.uniform(min_bound, max_bound, parameter_count))
    genome.strategy = [initial_sigma for _ in range(parameter_count)]
    return genome


def apply_self_adaptive_gaussian_mutation(genome):
    """
    Applies standard Evolutionary Programming mutation.
    The strategy variances adapt via log-normal perturbations alongside the objective variables.
    """
    dimensions = len(genome)
    learning_rate = 1.0 / np.sqrt(2.0 * dimensions)

    for i in range(dimensions):
        genome.strategy[i] = max(
            1e-5,
            genome.strategy[i] + random.gauss(0, learning_rate * genome.strategy[i])
        )
        genome[i] += random.gauss(0, genome.strategy[i])

    return genome,


def evaluate_genome_fitness(genome, u_signal, y_signal, na, nb, nc, nk):
    """
    Evaluates the MSE for a given candidate parameter set.
    """
    predicted_y, max_delay_lag = simulate_armax_system(u_signal, y_signal, genome, na, nb, nc, nk)

    # Assign a severe fitness penalty to dynamically unstable parameter sets to drive selection away from them.
    if not np.isfinite(predicted_y).all() or np.max(np.abs(predicted_y)) > 1e10:
        return (1e6,)

    try:
        mse_score = compute_mse(y_signal[max_delay_lag:], predicted_y[max_delay_lag:])
    except ValueError:
        return (1e6,)

    if np.isnan(mse_score) or np.isinf(mse_score) or mse_score > 1e6:
        return (1e6,)

    return (mse_score,)


def configure_ep_toolbox(hyperparameters):
    """
    Configures the DEAP toolbox environment.
    """
    ep_toolbox = base.Toolbox()
    total_parameters = hyperparameters["na"] + hyperparameters["nb"] + hyperparameters["nc"]

    thread_pool = multiprocessing.Pool()
    ep_toolbox.register("map", thread_pool.map)

    ep_toolbox.register("individual", initialize_ep_genome, creator.EPIndividual,
                        parameter_count=total_parameters, initial_sigma=hyperparameters.get("mutation_sigma", 0.5))
    ep_toolbox.register("population", tools.initRepeat, list, ep_toolbox.individual)
    ep_toolbox.register("mutate", apply_self_adaptive_gaussian_mutation)
    ep_toolbox.register("select", tools.selBest)

    return ep_toolbox, thread_pool


def execute_evolutionary_programming(u_train, y_train, hyperparameters):
    """
    Executes the generational optimization loop.
    Supports early stopping upon convergence and state persistence.
    """
    ep_toolbox, thread_pool = configure_ep_toolbox(hyperparameters)
    ep_toolbox.register("evaluate", evaluate_genome_fitness, u_signal=u_train, y_signal=y_train,
                        na=hyperparameters["na"], nb=hyperparameters["nb"],
                        nc=hyperparameters["nc"], nk=hyperparameters["nk"])

    ep_statistics = tools.Statistics(lambda genome: genome.fitness.values)
    ep_statistics.register("avg", np.mean)
    ep_statistics.register("min", np.min)

    try:
        checkpoint_path = hyperparameters.get("checkpoint_file")
        restored_state = load_checkpoint_state(checkpoint_path, config=hyperparameters)

        if restored_state:
            population = restored_state["population"]
            starting_generation = restored_state["generation"]
            hall_of_fame = restored_state["hall_of_fame"]
            ep_logbook = restored_state["logbook"]
        else:
            population = ep_toolbox.population(n=hyperparameters["pop_size"])
            starting_generation = 0
            hall_of_fame = tools.HallOfFame(1)
            ep_logbook = tools.Logbook()
            ep_logbook.header = ['gen', 'nevals'] + ep_statistics.fields

            fitness_scores = list(ep_toolbox.map(ep_toolbox.evaluate, population))
            for genome, score in zip(population, fitness_scores):
                genome.fitness.values = score

            hall_of_fame.update(population)
            ep_logbook.record(gen=0, nevals=len(population), **ep_statistics.compile(population))

        for generation in range(starting_generation + 1, hyperparameters["num_gen"] + 1):
            offspring_pool = [ep_toolbox.clone(genome) for genome in population]

            for offspring in offspring_pool:
                ep_toolbox.mutate(offspring)
                del offspring.fitness.values

            fitness_scores = list(ep_toolbox.map(ep_toolbox.evaluate, offspring_pool))
            for genome, score in zip(offspring_pool, fitness_scores):
                genome.fitness.values = score

            hall_of_fame.update(offspring_pool)

            # (μ + λ) selection strategy
            population[:] = ep_toolbox.select(population + offspring_pool, hyperparameters["pop_size"])

            ep_logbook.record(gen=generation, nevals=len(offspring_pool), **ep_statistics.compile(population))

            # Early stopping check: halt if improvement is negligible over 20 generations
            historical_mins = ep_logbook.select("min")
            if len(historical_mins) > 20 and abs(historical_mins[-1] - historical_mins[-21]) < 1e-6:
                print(f"[*] Optimization stopped early at generation {generation}.")
                if checkpoint_path:
                    save_checkpoint_state(population, generation, hall_of_fame, ep_logbook, checkpoint_path,
                                          config=hyperparameters)
                break

            if checkpoint_path and generation % hyperparameters["checkpoint_freq"] == 0:
                save_checkpoint_state(population, generation, hall_of_fame, ep_logbook, checkpoint_path,
                                      config=hyperparameters)

        return list(hall_of_fame[0]), ep_logbook

    finally:
        thread_pool.close()
        thread_pool.join()