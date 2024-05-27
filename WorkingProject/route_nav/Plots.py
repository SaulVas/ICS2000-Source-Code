import matplotlib.pyplot as plt
import numpy as np
import os
from PicturePoints import generate_picture_points
from Routing import cluster_picture_points, generate_optimal_routes


def plots(grid_width, grid_height, pic_points, labels, uav_routes):
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create a 1x3 subplot structure.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each subplot directly on the axes.
    plot_picture_points(ax1, grid_width, grid_height, pic_points)
    plot_clusters(ax2, grid_width, grid_height, pic_points, labels)
    plot_optimal_routes(ax3, grid_width, grid_height, uav_routes)

    plt.tight_layout()  # Adjust layout

    # Save each individual plot
    save_individual_plot(ax1, 'picture_points.png', results_dir)
    save_individual_plot(ax2, 'clusters.png', results_dir)
    save_individual_plot(ax3, 'optimal_routes.png', results_dir)

    # Save the combined plot
    combined_plot_path = os.path.join(results_dir, 'combined_plot.png')
    plt.savefig(combined_plot_path)
    plt.show()


def save_individual_plot(ax, filename, results_dir):
    # Create a new figure and axis to avoid saving the whole figure with subplots
    fig = plt.figure()
    new_ax = fig.add_subplot(111)
    for line in ax.get_lines():
        new_ax.plot(*line.get_data(), linestyle=line.get_linestyle(), marker=line.get_marker(), color=line.get_color())
    for collection in ax.collections:
        new_ax.scatter(*collection.get_offsets().T, color=collection.get_facecolor()[0])
    new_ax.set_xlim(ax.get_xlim())
    new_ax.set_ylim(ax.get_ylim())
    new_ax.set_title(ax.get_title())
    new_ax.set_xlabel(ax.get_xlabel())
    new_ax.set_ylabel(ax.get_ylabel())
    new_ax.grid(True)
    new_ax.legend(loc='best')
    fig.savefig(os.path.join(results_dir, filename))
    plt.close(fig)


def plot_picture_points(ax, grid_width, grid_height, pic_points):
    grid = np.zeros((grid_height, grid_width))
    for point in pic_points:
        grid[point[0], point[1]] = 1

    ax.imshow(grid, cmap='hot', interpolation='nearest', extent=[0, grid_width, 0, grid_height])
    ax.set_title('Picture Points on Grid')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
    for x, y in pic_points:
        ax.plot(y, x, 'bo')  # blue circle marker


def plot_clusters(ax, grid_width, grid_height, pic_points, labels):
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan']
    added_labels = set()

    for index, point in enumerate(pic_points):
        cluster_label = labels[index]
        label = f'Cluster {cluster_label}'
        if label not in added_labels:
            ax.scatter(point[1], point[0], color=colors[cluster_label % len(colors)], label=label)
            added_labels.add(label)
        else:
            ax.scatter(point[1], point[0], color=colors[cluster_label % len(colors)])

    ax.set_xlim([0, grid_width])
    ax.set_ylim([0, grid_height])
    ax.set_title('K-Means Clustering of Picture Points')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True)
    ax.legend()


def plot_optimal_routes(ax, grid_width, grid_height, routes):
    grid = np.zeros((grid_height, grid_width))
    ax.imshow(grid, cmap='gray', extent=[0, grid_width, 0, grid_height])

    for route in routes:
        path = np.array(route)
        ax.plot(path[:, 1], path[:, 0], marker='o')

    ax.set_title('Optimal Routes')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)

if __name__ == "__main__":
    # Grid dimensions
    width, height = 100, 100

    # Parameters
    NUM_UAVS = 3
    PARTITION_SIZE = 50

    ### PICTURE POINT GENERATION USING ILP
    picture_points = generate_picture_points(width, height, PARTITION_SIZE)
    picture_points = np.array(picture_points)

    ### CLUSTERING OF PICTURE POINTS
    point_labels = cluster_picture_points(NUM_UAVS, picture_points)

    ### ROUTE OPTIMIZATION USING GA
    routes = generate_optimal_routes(NUM_UAVS, point_labels, picture_points)

    plots(width, height, picture_points, point_labels, routes)
