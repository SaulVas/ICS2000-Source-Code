import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import sys
from Drones import Drone
from PicturePoints import generate_picture_points
from Routing import cluster_picture_points, generate_optimal_routes

def update_plot(frame, drones, car_positions, detected_cars, detected_positions, scat_drones, scat_cars, scat_detected_cars, scat_detection_positions):
    print(f"Frame {frame}: Updating positions...")
    drone_positions = [drone.get_position() for drone in drones]
    scat_drones.set_offsets(drone_positions)

    scat_cars.set_offsets(car_positions)

    for drone in drones:
        drone.update_position(car_positions, detected_positions, detected_cars)
        print(f"Drone at {drone.get_position()} with battery {drone.battery_life}")

    # Update scatter plots for detected cars and detection positions
    if detected_cars:
        scat_detected_cars.set_offsets(detected_cars)
    if detected_positions:
        scat_detection_positions.set_offsets(detected_positions)

    if len(detected_cars) == len(car_positions):
        print("All cars detected. Stopping the simulation.")
        sys.exit()

    return scat_drones, scat_cars, scat_detected_cars, scat_detection_positions

def simulate_drones_visual(drones, car_positions, detected_cars, detected_positions, grid_width, grid_height):
    fig, ax = plt.subplots()
    # Setting the limits of the plot
    ax.set_xlim(0, grid_width)
    ax.set_ylim(0, grid_height)

    # Plotting drones and cars
    drone_positions = [drone.get_position() for drone in drones]
    if drone_positions:
        scat_drones = ax.scatter(*zip(*drone_positions), c='blue', label='Drones')
    else:
        scat_drones = ax.scatter([], [], c='blue', label='Drones')
    
    if car_positions:
        scat_cars = ax.scatter(*zip(*car_positions), c='red', label='Cars')
    else:
        scat_cars = ax.scatter([], [], c='red', label='Cars')

    # Initialize empty scatter plots for detected cars and detection positions
    scat_detected_cars = ax.scatter([], [], c='green', label='Detected Cars')
    scat_detection_positions = ax.scatter([], [], c='black', label='Detection Positions')

    animation = FuncAnimation(fig, update_plot,
                              fargs=(drones, car_positions, detected_cars, detected_positions, scat_drones, scat_cars, scat_detected_cars, scat_detection_positions),
                              frames=None, interval=100, repeat=True)

    plt.legend()
    plt.show()
    return animation

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
    NUM_UAVS = 3
    width, height = 100, 100
    PARTITION_SIZE = 50
    home_pos = (50, 0)
    num_cars = 10

    picture_points = generate_picture_points(width, height, PARTITION_SIZE)
    picture_points = np.array(picture_points)

    point_labels = cluster_picture_points(NUM_UAVS, picture_points)
    routes = generate_optimal_routes(NUM_UAVS, point_labels, picture_points)

    home_pos = (50, 0)

    drones = [Drone(home_pos, route) for route in routes]
    car_positions = [(random.uniform(0, width), random.uniform(0, height)) for _ in range(num_cars)]
    detected_car_positions = []
    detected_positions = []

    simulate_drones_visual(drones, car_positions, detected_car_positions, detected_positions, width, height)
