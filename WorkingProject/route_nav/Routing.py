from sklearn.cluster import KMeans
from deap import creator, base, tools, algorithms
import random
import numpy as np

def cluster_picture_points(num_uavs, picture_points):
    kmeans = KMeans(n_clusters=num_uavs)
    kmeans.fit(picture_points)
    labels = kmeans.labels_

    return labels

def find_optimal_route(coordinates, population_size=100, generations=400, cxpb=0.7, mutpb=0.2, seed=42):
    """
    Find the optimal route through a list of xy coordinates using a genetic algorithm.
    
    Parameters:
    - coordinates: List of (x, y) tuples representing the coordinates.
    - population_size: Size of the population (default is 100).
    - generations: Number of generations (default is 400).
    - cxpb: Probability of crossover (default is 0.7).
    - mutpb: Probability of mutation (default is 0.2).
    - seed: Random seed for reproducibility (default is 42).

    Returns:
    - best_individual: The best route found as a list of indices.
    - best_route: The best route found as a list of coordinates.
    - best_distance: The total distance of the best route.
    """

    def distance(point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def eval_route(individual):
        total_distance = 0
        for i in range(len(individual)):
            total_distance += distance(coordinates[individual[i-1]], coordinates[individual[i]])
        return total_distance,

    toolbox = base.Toolbox()

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox.register("indices", random.sample, range(len(coordinates)), len(coordinates))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", eval_route)

    random.seed(seed)
    
    population = toolbox.population(n=population_size)
    
    algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, verbose=False)
    
    best_individual = tools.selBest(population, k=1)[0]
    best_route = [coordinates[i] for i in best_individual]

    return best_route

def generate_optimal_routes(num_uavs, labels, picture_points):
    routes = []
    for label in range(num_uavs):
        cluster_points = [picture_points[idx] for idx, lab in zip(range(len(picture_points)),
                                                                  labels) if lab == label]
        if cluster_points:
            best_route = find_optimal_route(cluster_points)
            routes.append(best_route)

    return routes
