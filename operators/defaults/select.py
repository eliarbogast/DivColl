import numpy as np, os
from utils.optim_utils import get_invalid_robots
from utils.disk_utils import load_pop, save_pop
from scipy.spatial.distance import cdist

def calculate_local_competition(fitness, descriptor, population_descriptors, population_fitness, k=5):
    """
        Calculate local competition score for an individual:
        - Find k nearest neighbors in descriptor space within the population
        - Score is fraction of neighbors with fitness lower than given fitness
        """
    distances = cdist([descriptor], population_descriptors)[0]
    nearest_indices = np.argsort(distances)[:k]
    return np.sum(fitness > population_fitness[nearest_indices]) / k

def get_best(pop_fitness, offspring_fitness, descriptors):
    """
        Select best individuals between population and offspring:
        - Identify invalid individuals in both sets
        - Combine fitnesses and compute local competition scores for all
        - Rank by product of fitness and local competition, descending
        - Filter out invalids and keep top individuals equal in number to original population
        """
    invalid_pop = get_invalid_robots(pop_fitness)
    invalid_offspring = get_invalid_robots(offspring_fitness) + pop_fitness.shape[0]
    invalid = np.concatenate((invalid_pop, invalid_offspring))
    fitness = np.concatenate((pop_fitness, offspring_fitness))
    local_competition_scores = np.array([calculate_local_competition(f, b, descriptors, fitness.max(1))
                                         for f, b in zip(fitness.max(1), descriptors)])
    best_idx = (fitness.max(1) * local_competition_scores).argsort()[::-1]
    best_idx = np.array([i for i in best_idx if i not in invalid])
    best_idx = best_idx[:pop_fitness.shape[0]]
    best_fitness = fitness[best_idx]
    return best_idx, best_fitness

def select(pop_file, pop_fitness, offspring_file, offspring_fitness, outdir):
    """
        Combine population and offspring, select best individuals by fitness and local competition,
        build new population and save to disk.

        Returns new population file path and best fitness array.
        """
    pop = load_pop(pop_file)
    offspring = load_pop(offspring_file)
    combined_pop = {
        "body_geno": pop["body_geno"] + offspring["body_geno"],
        "spring_geno": pop["spring_geno"] + offspring["spring_geno"],
        "id": pop["id"] + offspring["id"],
        "descriptor": pop["descriptor"] + offspring["descriptor"]
    }
    descriptors = np.array([x for x in combined_pop["descriptor"]])
    best_idx, best_fitness = get_best(pop_fitness, offspring_fitness, descriptors)
    new_pop = {
        "body_geno": [],
        "spring_geno": [],
        "points": [],
        "springs": [],
        "id": []
    }
    n = len(pop["body_geno"])
    for idx in best_idx:
        if idx < n:
            new_pop["body_geno"].append(pop["body_geno"][idx])
            new_pop["spring_geno"].append(pop["spring_geno"][idx])
            new_pop["points"].append(pop["points"][idx])
            new_pop["springs"].append(pop["springs"][idx])
            new_pop["id"].append(pop["id"][idx])
        else:
            new_pop["body_geno"].append(offspring["body_geno"][idx-n])
            new_pop["spring_geno"].append(offspring["spring_geno"][idx-n])
            new_pop["points"].append(offspring["points"][idx-n])
            new_pop["springs"].append(offspring["springs"][idx-n])
            new_pop["id"].append(offspring["id"][idx-n])
    pop_outfile = os.path.join(outdir, "robots.pkl")
    save_pop(new_pop, pop_outfile)
    return pop_outfile, best_fitness