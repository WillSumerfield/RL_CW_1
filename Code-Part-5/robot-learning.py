#########################
# DO NOT EDIT THIS FILE #
#########################

# Imports from external libraries
import time
import numpy as np
import pygame

# Imports from this project
from environment import Environment
from robot import Robot
from graphics import Graphics
from demonstrator import Demonstrator


# Set the numpy random seed
seed = int(time.time())
np.random.seed(seed)
# Initialize Pygame
pygame.init()
# Create an environment (the "physical" world)
environment = Environment()
# Create a robot (the robot's "brain" making the decisions)
robot = Robot(environment.forward_kinematics)
# Create a graphics object (this will create a window and draw on the window)
graphics = Graphics()
# Create a demonstrator (the "human" who will provide the demonstrations)
demonstrator = Demonstrator(environment)
# Give the robot some demonstrations
robot.get_demos(demonstrator)

# Main loop
running = True
while running:
    # Check for any user input
    for event in pygame.event.get():
        # Closing the window
        if event.type == pygame.QUIT:
            running = False
    # Robot selects an action, and decides whether it has finished the episode
    curr_state = np.copy(environment.state)
    action, episode_done = robot.select_action(curr_state)
    # If the episode has finished, create a new environment
    if episode_done:
        environment = Environment()
        demonstrator = Demonstrator(environment)
        running = robot.reset()
        if not running:
            break
        robot.get_demos(demonstrator)
    # If the episode has not finished, execute the selected action
    else:
        next_state = environment.step(action)
    # Draw the environment, and any visualisations, on the window
    graphics.draw(environment, robot.planning_visualisation_lines, robot.policy_visualisation_lines)

# If we have broken out of the main loop, quite pygame and end the program
pygame.quit()
