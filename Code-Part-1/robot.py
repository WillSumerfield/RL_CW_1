####################################
#      YOU MAY EDIT THIS FILE      #
# MOST OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np

# Imports from this project
import constants
import config


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self):
        # The environment
        self.environment = None
        # A list of visualisations which will be displayed on the right-side of the window
        self.visualisation_lines = []
        self.visualisation_circles = []

    # Reset the robot at the start of an episode
    def reset(self):
        pass

    # Give the robot access to the environment's dynamics
    def give_environment_access(self, environment):
        self.environment = environment

    # Function to get the next action in the plan
    def select_action(self, state):
        # Initially, we just return a random action, but you should implement a planning algorithm
        action = np.random.uniform(low=-constants.MAX_ACTION_MAGNITUDE, high=constants.MAX_ACTION_MAGNITUDE, size=2)
        episode_done = False
        return action, episode_done


# The VisualisationLine class enables us to store a line segment which will be drawn to the screen
class VisualisationLine:
    # Initialise a new visualisation (a new line)
    def __init__(self, x1, y1, x2, y2, colour=(255, 255, 255), width=0.01):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.colour = colour
        self.width = width


# The VisualisationCircle class enables us to a circle which will be drawn to the screen
class VisualisationCircle:
    # Initialise a new visualisation (a new circle)
    def __init__(self, x, y, radius, colour=(255, 255, 255)):
        self.x = x
        self.y = y
        self.radius = radius
        self.colour = colour