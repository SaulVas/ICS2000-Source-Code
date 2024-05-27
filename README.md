# ICS2000-GAPT Drone Follower Project

## Introduction

Our project specifically focuses on utilizing three UAVs for efficient and effective search and rescue missions. The primary objective is to interface with these drones to survey extensive areas and identify stationary objects, simulating a scenario where UAVs can locate and identify stranded vehicles in the aftermath of natural disasters or accidents. For our proof of concept, we selected cars as the stationary objects of interest.

For our project, we were given three DJI Mini 3 drones, which have advanced capabilities and suitability for precise aerial surveys. The project was divided into three core sections: pathfinding and route optimization, object detection, and drone communication and simulation. Each section addresses a fundamental aspect of the overall mission, ensuring a comprehensive and functional solution.

Ensuring effective pathfinding and route optimization to cover large areas thoroughly and as quickly as possible is a critical component of our project. For this, we used genetic algorithms and k-means clustering. By using these AI techniques, it is possible to ensure that the surveying process is carried out as effectively as possible.

The majority of search and rescue operations depend heavily on object detection, which is the subject of the second portion of our project. The system's ability to correctly identify cars from aerial imagery is the goal here. To do this, we trained the YOLOv5m model on a public online dataset [1].

## Source Code Structure

The Source Code directory contains 2 directories:

1. MSDKSample3 -> Contains our failed implementation of an Android app, however, it is an empty project that has the mobile SDK integrated with it.
2. WorkingProject -> Contains our Python project implementing various AI techniques, simulations and evaluations.

## Packages and Requirements

To run any code within the working project Directory one must follow the steps outlined below:
1. Create a python venv.
2. run the following command from within the WorkingProject directory:
```bash
pip install -r requirements.txt
```

## Proof of concept

1. A demonstration of our object detection model on real-life images taken using the drones supplied. These images would contain different types of cars, different backgrounds, and the cars placed in different positions in the frame. This can be run by running *WorkingProject/obj/detect.py* from within the *obj* directory.
2. Creating a visual simulation of three drones surveying a 1000m by 1000m grid. All the values used in the simulation are based on the specs listed on DJI's website. The visual simulation can be viewed by running *WorkingProject/route_nav/Simulation.py* from within the *route_nav* directory.
