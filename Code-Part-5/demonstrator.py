####################################
#      YOU MAY EDIT THIS FILE      #
# MOST OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np

# Imports from this project
import constants


# The Demonstrator class is the "human" which provides the demonstrations to the robot
class Demonstrator:

    # Initialise a new demonstrator
    def __init__(self, environment):
        # The initial state which the demonstration starts from
        self.init_state = environment.init_state
        # The demonstrator has access to the true environment dynamics and the goal state
        self.dynamics = environment.dynamics
        self.goal_state = environment.goal_state
        # The forward kinematics
        self.forward_kinematics = environment.forward_kinematics

    # Generate a demonstration using the cross-entropy method
    def generate_demonstration(self):
        # Create some placeholders for the data
        sampled_actions = np.zeros([constants.CEM_NUM_ITER, constants.CEM_NUM_PATHS, constants.CEM_PATH_LENGTH, 2], dtype=np.float32)
        sampled_paths = np.zeros([constants.CEM_NUM_ITER, constants.CEM_NUM_PATHS, constants.CEM_PATH_LENGTH + 1, 2], dtype=np.float32)
        path_distances = np.zeros([constants.CEM_NUM_ITER, constants.CEM_NUM_PATHS], dtype=np.float32)
        action_mean = np.zeros([constants.CEM_NUM_ITER, constants.CEM_PATH_LENGTH, 2], dtype=np.float32)
        action_std = np.zeros([constants.CEM_NUM_ITER, constants.CEM_PATH_LENGTH, 2], dtype=np.float32)
        # Set the current state
        state = self.init_state
        # Loop over the CEM iterations
        for iter_num in range(constants.CEM_NUM_ITER):
            # Loop over all the paths that will be sampled
            for path_num in range(constants.CEM_NUM_PATHS):
                # The start of each path is the robot's current state
                sampled_paths[iter_num, path_num, 0] = state
                curr_state = state
                path_collision = False
                # Sample actions for each step of the episode
                for step in range(constants.CEM_PATH_LENGTH):
                    # If this is the first iteration, then sample a uniformly random action
                    if iter_num == 0:
                        action = np.random.uniform(low=-constants.MAX_ACTION_MAGNITUDE, high=constants.MAX_ACTION_MAGNITUDE, size=2)
                    # If this is not the first iteration, then sample an action from the distribution calculated in the previous iteration
                    else:
                        action = np.random.normal(loc=action_mean[iter_num-1, step], scale=action_std[iter_num-1, step])
                        # We need to clip this action because the normal distribution is unbounded
                        action = np.clip(action, a_min=-constants.MAX_ACTION_MAGNITUDE, a_max=constants.MAX_ACTION_MAGNITUDE)
                    # Calculate the next state using the environment dynamics
                    next_state, collision = self.dynamics(curr_state, action)
                    if collision:
                        path_collision = True
                    # Populate the placeholders with the action and next state
                    sampled_actions[iter_num, path_num, step] = action
                    sampled_paths[iter_num, path_num, step + 1] = next_state
                    # Update the state in this planning path
                    curr_state = next_state
                # Calculate the distance between the final state and the goal, for this path
                final_hand_pos = self.forward_kinematics(next_state)[2]
                distance = np.linalg.norm(self.goal_state - final_hand_pos)
                # If there was a collision, penalise this demonstration
                if path_collision:
                    distance += 1.0
                path_distances[iter_num, path_num] = distance
            # Find the elite paths, which we do here by getting the paths with the minimum distance to the goal
            elites = np.argsort(path_distances[iter_num])[:constants.CEM_NUM_ELITES]
            # Use the elite paths to update the action distribution
            elite_actions = sampled_actions[iter_num, elites]
            action_mean[iter_num] = np.mean(elite_actions, axis=0)
            action_std[iter_num] = np.std(elite_actions, axis=0)
        # The demonstration is the best path after the iterations
        best_path = np.argmin(path_distances[-1])
        # Now create the state-action pairs
        demonstration = []
        for step in range(constants.CEM_PATH_LENGTH):
            state = sampled_paths[-1, best_path, step]
            action = sampled_actions[-1, best_path, step]
            demonstration.append((state, action))
        return demonstration
