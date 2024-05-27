import numpy as np

BATTERY_LIFE = 2400  # 40 minutes in seconds

class Drone:
    def __init__(self, home, route=None):
        self.home = np.array(home)
        self.current_pos = np.array(home)
        self.in_flight = True
        self.battery_life = BATTERY_LIFE
        self.speed = 0.55  # speed in meters per second
        self.route = [np.array(point) for point in route] if route else []
        self.route_index = 0
        self.original_route = None  # Store original route if returning home
        self.original_index = 0
        self.safety_margin = 30
        self.returning_home = False  # Flag to indicate returning home

    def update_position(self, car_positions, detected_positions, detected_cars):
        if not self.in_flight:
            return  # Stops moving if landed

        self.battery_life -= 1
        if self.route_index < len(self.route):
            current_point = self.current_pos
            next_point = self.route[self.route_index]
            direction = next_point - current_point
            distance = np.linalg.norm(direction)

            if distance > 0:
                direction = direction / distance  # Normalize the direction vector

            travel_distance = min(self.speed, distance)
            self.current_pos = current_point + direction * travel_distance

            if np.linalg.norm(self.current_pos - next_point) < 1e-2:
                self.current_pos = next_point  # Snap to the waypoint to avoid floating-point issues
                self.detect_car(car_positions, detected_positions, detected_cars)
                self.route_index += 1
                if self.route_index == len(self.route):
                    if self.returning_home:
                        self.refuel()
                    else:
                        self.route_index = 0

        if (not self.returning_home
            and self.calculate_return_home_battery() + self.safety_margin > self.battery_life):
            self.return_to_home()

    def detect_car(self, car_positions, detected_positions, detected_cars):
        for car_pos in car_positions:
            if (abs(self.current_pos[0] - car_pos[0]) <= 4
                and abs(self.current_pos[1] - car_pos[1]) <= 5):
                if car_pos not in detected_cars:
                    detected_cars.append(car_pos)  # Add detected car position to the list
                    detected_positions.append(self.current_pos.copy())

    def calculate_return_home_battery(self):
        home_distance = np.linalg.norm(self.home - self.current_pos)
        return home_distance / self.speed  # Time in seconds to return home

    def return_to_home(self):
        self.original_route = self.route  # Save the original route
        self.route = [self.home]  # Set route to only include home
        self.original_index = self.route_index
        self.route_index = 0
        self.returning_home = True

    def refuel(self):
        print("Refueling...")
        self.battery_life = BATTERY_LIFE  # Reset battery life
        self.route = self.original_route if self.original_route else [self.home]
        self.route_index = self.original_index
        self.returning_home = False
        self.in_flight = True

    def start_mission(self):
        self.in_flight = True

    def get_position(self):
        return self.current_pos.tolist()  # Convert numpy array to list for consistency

    def reset(self):
        self.current_pos = self.home
        self.route_index = 0
        self.battery_life = BATTERY_LIFE
        self.returning_home = False
        self.original_route = None
        self.original_index = 0
