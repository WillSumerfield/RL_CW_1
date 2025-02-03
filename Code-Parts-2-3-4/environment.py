#########################
# DO NOT EDIT THIS FILE #
#########################

# Imports from external libraries
import numpy as np

# Imports from this project
import constants
import config


# The environment class defines where the robot starts, where the goal is, and where the obstacle is.
class Environment:

    # Initialisation of a new environment
    def __init__(self):
        # Same environment for each episode
        # Obstacle position
        self.obstacle_pos = np.array(constants.OBSTACLE_POS)
        self.obstacle_radius = constants.OBSTACLE_RADIUS
        # Initial state
        self.init_state = np.array(constants.INIT_STATE)
        # Position of the robot's base
        self.robot_base_pos = np.array(constants.ROBOT_BASE_POS)
        # Goal state
        self.goal_state = constants.GOAL_STATE
        # Set the current state to the initial state
        self.state = self.init_state

    # Reset the environment, i.e. set the state to the initial state
    def reset(self):
        self.state = self.init_state

    # Step the environment by executing one action
    def step(self, action):
        # Update the state
        next_state = self.dynamics(self.state, action)
        self.state = next_state
        # Return the next state
        return next_state

    # The environment dynamics, i.e. the transition function
    def dynamics(self, state, action):
        # First, clip the action in each dimension
        action = np.clip(action, -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
        # In this environment, the change in state (joint angles) is 3 x the action
        next_state = state + 3 * action
        # Get the joint positions for this next state
        next_joint_pos = self.forward_kinematics(next_state)
        # Check if the robot is inside the obstacle
        intersection_1 = self.line_circle_intersection(next_joint_pos[0], next_joint_pos[1], self.obstacle_pos, self.obstacle_radius)
        intersection_2 = self.line_circle_intersection(next_joint_pos[1], next_joint_pos[2], self.obstacle_pos, self.obstacle_radius)
        if intersection_1 or intersection_2:
            # If there is an intersection, then the state remains as it was before
            next_state = state
        return next_state

    # The robot's forward kinematics
    def forward_kinematics(self, state):
        # Joint 1 position
        joint1_x = self.robot_base_pos[0]
        joint1_y = self.robot_base_pos[1] + constants.ROBOT_LINK_LENGTHS[0]
        joint1_pos = np.array([joint1_x, joint1_y])
        # Joint 2 position
        joint2_x = joint1_x - constants.ROBOT_LINK_LENGTHS[1] * np.sin(state[0])
        joint2_y = joint1_y + constants.ROBOT_LINK_LENGTHS[1] * np.cos(state[0])
        joint2_pos = np.array([joint2_x, joint2_y])
        # Hand position
        hand_x = joint2_x - constants.ROBOT_LINK_LENGTHS[2] * np.sin(state[0] + state[1])
        hand_y = joint2_y + constants.ROBOT_LINK_LENGTHS[2] * np.cos(state[0] + state[1])
        hand_pos = np.array([hand_x, hand_y])
        # Return the positions
        return [joint1_pos, joint2_pos, hand_pos]

    # Function to check if a line intersects a circle
    def line_circle_intersection(self, p1, p2, center, radius):
        (x1, y1) = p1
        (x2, y2) = p2
        (xc, yc) = center
        # Compute the differences
        dx = x2 - x1
        dy = y2 - y1
        # Quadratic coefficients
        A = dx ** 2 + dy ** 2
        B = 2 * (dx * (x1 - xc) + dy * (y1 - yc))
        C = (x1 - xc) ** 2 + (y1 - yc) ** 2 - radius ** 2
        # Calculate discriminant
        discriminant = B ** 2 - 4 * A * C
        # Check for intersection
        if discriminant < 0:
            # No intersection
            return None
        elif discriminant == 0:
            # One intersection (tangent)
            t = -B / (2 * A)
            if 0 <= t <= 1:
                intersection = (x1 + t * dx, y1 + t * dy)
                return intersection
            else:
                return None
        else:
            # Two intersections
            sqrt_discriminant = np.sqrt(discriminant)
            t1 = (-B - sqrt_discriminant) / (2 * A)
            t2 = (-B + sqrt_discriminant) / (2 * A)
            # Initialize list to hold valid t values
            t_values = []
            if 0 <= t1 <= 1:
                t_values.append(t1)
            if 0 <= t2 <= 1:
                t_values.append(t2)
            if not t_values:
                # Intersections are not on the segment
                return None
            # Find the smallest t (nearest to p1)
            t_nearest = min(t_values)
            intersection = (x1 + t_nearest * dx, y1 + t_nearest * dy)
            return intersection
