import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from collections import Counter
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import numpy as np
from PicturePoints import generate_picture_points
from Drones import Drone

def evaluate_pic_points(grid_width, grid_height, partition_size=1):
    start_time = time.time()
    picture_points = generate_picture_points(grid_width, grid_height, partition_size)
    end_time = time.time()
    
    coverage = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
    
    for (i, j) in picture_points:
        for di in range(-3, 4):
            for dj in range(-4, 5):
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_height and 0 <= nj < grid_width:
                    coverage[ni][nj] += 1
    
    total_cells = grid_width * grid_height
    covered_cells = sum(1 for row in coverage for cell in row if cell > 0)
    redundant_coverage = sum(cell - 1 for row in coverage for cell in row if cell > 1)
    
    coverage_percentage = covered_cells / total_cells * 100
    redundancy_percentage = redundant_coverage / total_cells * 100
    time_taken = end_time - start_time
    
    print(f"Coverage Percentage: {coverage_percentage:.2f}%")
    print(f"Redundancy Percentage: {redundancy_percentage:.2f}%")
    print(f"Time Taken: {time_taken:.2f} seconds")
    
    return {
        "coverage_percentage": coverage_percentage,
        "redundancy_percentage": redundancy_percentage,
        "time_taken": time_taken,
        "picture_points": picture_points
    }

def evaluate_clustering(num_uavs, picture_points):
    kmeans = KMeans(n_clusters=num_uavs, random_state=42)
    start_time = time.time()
    kmeans.fit(picture_points)
    end_time = time.time()
    
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(picture_points, labels)
    davies_bouldin = davies_bouldin_score(picture_points, labels)
    
    # Balance metric: calculate the distribution of points in clusters
    label_counts = Counter(labels)
    balance = np.std(list(label_counts.values())) / np.mean(list(label_counts.values()))
    
    clustering_time = end_time - start_time
    
    print(f"Silhouette Score: {silhouette_avg:.2f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")
    print(f"Balance: {balance:.2f}")
    print(f"Clustering Time: {clustering_time:.2f} seconds")
    
    return {
        "silhouette_score": silhouette_avg,
        "davies_bouldin_index": davies_bouldin,
        "balance": balance,
        "clustering_time": clustering_time,
        "labels": labels
    }

def evaluate_genetic_algorithm(coordinates, label, population_size=100, generations=400, cxpb=0.7, mutpb=0.2, seed=42):
    """
    Evaluate the performance of a genetic algorithm in finding the optimal route through a list of xy coordinates.
    
    Parameters:
    - coordinates: List of (x, y) tuples representing the coordinates.
    - population_size: Size of the population (default is 100).
    - generations: Number of generations (default is 400).
    - cxpb: Probability of crossover (default is 0.7).
    - mutpb: Probability of mutation (default is 0.2).
    - seed: Random seed for reproducibility (default is 42).

    Returns:
    - evaluation_results: A dictionary containing evaluation metrics and other relevant information.
    """
    
    def distance(point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def eval_route(individual):
        total_distance = 0
        for i in range(len(individual)):
            total_distance += distance(coordinates[individual[i-1]], coordinates[individual[i]])
        return total_distance,

    # Setup genetic algorithm
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
    
    # Run genetic algorithm and track best distances
    best_distances = []
    start_time = time.time()
    
    for gen in range(generations):
        population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
        fits = toolbox.map(toolbox.evaluate, population)
        for fit, ind in zip(fits, population):
            ind.fitness.values = fit
        population = toolbox.select(population, k=len(population))
        
        best_individual = tools.selBest(population, k=1)[0]
        best_distance = eval_route(best_individual)[0]
        best_distances.append(best_distance)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    best_individual = tools.selBest(population, k=1)[0]
    best_distance = eval_route(best_individual)[0]
    best_route = [coordinates[i] for i in best_individual]
    
    # Plotting the convergence
    plt.figure(figsize=(10, 6))
    plt.plot(best_distances, label='Best Distance per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.title('Genetic Algorithm Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/GA_convergence_{label}.png')

    print(f"Best Distance: {best_distance}")
    print(f"Time Taken: {total_time} seconds")
    
    # Return evaluation results
    evaluation_results = {
        "best_individual": best_individual,
        "best_route": best_route,
        "best_distance": best_distance,
        "total_time": total_time,
        "best_distances": best_distances
    }
    
    return evaluation_results

def simulate_drones(drones, car_positions, detected_cars, detected_positions):
    frame = 0
    while True:
        for drone in drones:
            drone.update_position(car_positions, detected_positions, detected_cars)
        
        if len(detected_cars) == len(car_positions):
            break

        frame += 1

    return frame

if __name__ == "__main__":
    # simulation variables
    width, height = 100, 100
    PARTITION_SIZE = 50
    home_pos = (50, 0)
    num_uavs = 3
    num_cars = 10

    # picture points testing
    results = evaluate_pic_points(grid_width=width, grid_height=height, partition_size=PARTITION_SIZE)
    picture_points = results["picture_points"]

    # clustering testing
    clustering_results = evaluate_clustering(num_uavs, picture_points)
    labels = clustering_results["labels"]

    # GA testing
    GA_results = []
    for label in range(num_uavs):
        cluster_points = [picture_points[idx] for idx, lab in zip(range(len(picture_points)),
                                                                  labels) if lab == label]
        if cluster_points:
            best_route = evaluate_genetic_algorithm(cluster_points, label)
            GA_results.append(best_route)

    routes = [result["best_route"] for result in GA_results]

    # sim testing
    drones = [Drone(home_pos, route) for route in routes]
    simulation_times = []
    for index in range(10000):
        car_positions = [(random.uniform(0, width), random.uniform(0, height)) for _ in range(num_cars)]
        detected_car_positions = []
        detected_positions = []
        frame = simulate_drones(drones, car_positions, detected_car_positions, detected_positions)
        simulation_times.append(frame)
        print(f"Simulation {index + 1} took {frame:.2f} seconds")

    simulation_times_np = np.array(simulation_times)
    mean_time = np.mean(simulation_times_np)
    median_time = np.median(simulation_times_np)
    std_time = np.std(simulation_times_np)
    min_time = np.min(simulation_times_np)
    max_time = np.max(simulation_times_np)

    print(f"\nMean time: {mean_time:.2f} seconds")
    print(f"Median time: {median_time:.2f} seconds")
    print(f"Standard Deviation: {std_time:.2f} seconds")
    print(f"Min time: {min_time:.2f} seconds")
    print(f"Max time: {max_time:.2f} seconds")

    print(f"\nILP Picture Points Problem Results:")
    print(f"Coverage Percentage: {results['coverage_percentage']:.2f}%")
    print(f"Redundancy Percentage: {results['redundancy_percentage']:.2f}%")
    print(f"Time Taken: {results['time_taken']:.2f} seconds")

    print(f"\nClustering Results:")
    print(f"Silhouette Score: {clustering_results['silhouette_score']:.2f}")
    print(f"Davies-Bouldin Index: {clustering_results['davies_bouldin_index']:.2f}")
    print(f"Balance: {clustering_results['balance']:.2f}")
    print(f"Clustering Time: {clustering_results['clustering_time']:.2f} seconds")

    print(f"\nGenetic Algorithm Results:")
    for idx, result in enumerate(GA_results):
        print(f"Cluster {idx} - Best Distance: {result['best_distance']:.2f} - Time Taken: {result['total_time']:.2f} seconds")

    # Plotting boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(simulation_times, vert=False, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))

    plt.title('Boxplot of Simulation Times')
    plt.ylabel('Simulations')
    plt.xlabel('Time (seconds)')
    plt.savefig('results/boxplot_simulation_times.png')
    plt.show()