####################################
#      YOU MAY EDIT THIS FILE      #
# MOST OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

# Imports from this project
import constants
import config

# Configure matplotlib for interactive mode
plt.ion()


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self, forward_kinematics):
        # Give the robot the forward kinematics function, to calculate the hand position from the state
        self.forward_kinematics = forward_kinematics
        # A list of visualisations which will be displayed on the right-side of the window
        self.planning_visualisation_lines = []
        self.model_visualisation_lines = []
        # The position of the robot's base
        self.robot_base_pos = np.array(constants.ROBOT_BASE_POS)
        # The goal state
        self.goal_state = 0
        # The number of steps currently taken in the episode
        self.episode_steps = 0
        # The number of episodes currently taken in the dynamics data collection
        self.dynamics_episode_count = 0
        # The data collected for the dynamics model
        self.dynamics_model_data = np.zeros((config.DYNAMICS_DATA_COLLECTION_EPISODE_LENGTH*config.DYNAMICS_DATA_COLLECTION_EPISODES, 3, 2))
        # The dynamics model
        self.dynamics_model = DynamicsModel()
        # The CEM distribution
        self.action_distribution = None
        # The CEM path
        self.path = None
        self.test = True

    # Reset the robot at the start of an episode
    def reset(self):
        self.episode_steps = 0
        self.model_visualisation_lines = []
        self.path = None
        self.dynamics_model_data = np.zeros((config.DYNAMICS_DATA_COLLECTION_EPISODE_LENGTH*config.DYNAMICS_DATA_COLLECTION_EPISODES, 3, 2))

    # Give the robot access to the goal state
    def set_goal_state(self, goal_state):
        self.goal_state = goal_state

    # Function to get the next action in the plan
    def select_action(self, state):

        # At first, explore to train the dynamics model.
        if self.dynamics_episode_count < config.DYNAMICS_DATA_COLLECTION_EPISODES:
            action = np.random.uniform(low=-constants.MAX_ACTION_MAGNITUDE, high=constants.MAX_ACTION_MAGNITUDE, size=2) # Randomly explore
            if self.episode_steps+1 >= config.DYNAMICS_DATA_COLLECTION_EPISODE_LENGTH:
                self.dynamics_episode_count += 1
                return action, True
        
        # Train the dynamics model if enough data has been collected
        else:
            config.FRAME_RATE = 10
            if self.test:
                self.test = False
                self.train_dynamic_model()

            # After training the dynamics model, use the cross-entropy method to plan
            if self.path is None:
                self.create_plan(state)

            action = self.path[self.episode_steps, 1]

            # Visualise the action
            state_action = torch.cat((torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)))
            pred_state = self.dynamics_model(state_action).detach().numpy()
            pred_pos = self.forward_kinematics(pred_state)
            state_pos = self.forward_kinematics(state)
            v1 = VisualisationLine(state_pos[2][0], state_pos[2][1], pred_pos[2][0], pred_pos[2][1], (0, 255, 0), 0.005)
            self.model_visualisation_lines += [v1]

        self.episode_steps += 1
        episode_done = self.episode_steps >= config.CEM_PATH_LENGTH

        return action, episode_done

    def train_dynamic_model(self):
        self.dynamics_model.train()
        optimizer = optim.Adam(self.dynamics_model.parameters())
        criterion = nn.MSELoss()

        # A list of the loss for each training step
        losses = []

        # Train until the loss is below a threshold
        epoch = 0
        while True:
            # Get a random batch of data
            batch = self.dynamics_model_data[np.random.choice(self.dynamics_model_data.shape[0], config.DYNAMICS_BATCH_SIZE)]
            states = torch.tensor(batch[:,0], dtype=torch.float32)
            actions = torch.tensor(batch[:,1], dtype=torch.float32)
            next_states = torch.tensor(batch[:,2], dtype=torch.float32)

            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            predicted_next_states = self.dynamics_model(torch.cat((states, actions), 1))
            # Compute the loss
            loss = criterion(predicted_next_states, next_states)
            # Backward pass
            loss.backward()
            # Optimize
            optimizer.step()

            losses.append(loss.item())

            epoch += 1
            if epoch % 50 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

            # Check if the loss is below the threshold
            if loss < config.DYNAMICS_TRAINING_THRESHOLD:
                break

        self.dynamics_model.eval()

        # Plot the loss
        if True:
            plt.plot(losses)
            plt.title("Dynamics Model Loss")
            plt.xlabel("Batch Number")
            plt.ylabel("Loss")
            plt.show()
            plt.savefig("loss.png")

    # Function to add a transition to the buffer
    def add_transition(self, state, action, next_state):
        if self.dynamics_episode_count >= config.DYNAMICS_DATA_COLLECTION_EPISODES:
            return
        total_steps = self.dynamics_episode_count*config.DYNAMICS_DATA_COLLECTION_EPISODE_LENGTH + self.episode_steps
        self.dynamics_model_data[total_steps] = np.array([state, action, next_state])

    # Create a new path plan using our dymamics model
    def create_plan(self, state):

        # Iteratively improve the plans
        for itr in range(config.CEM_ITERATIONS):

            # Get a series of plans
            paths =  np.zeros((config.CEM_PATHS, config.CEM_PATH_LENGTH, 2, 2))
            for p in range(config.CEM_PATHS):
                cur_state = state

                # Generate a path
                for step in range(config.CEM_PATH_LENGTH):

                    # If this is our first pass, use a normal distribution
                    if self.action_distribution is None:
                        action = np.random.uniform(low=-constants.MAX_ACTION_MAGNITUDE, high=constants.MAX_ACTION_MAGNITUDE, size=2)
                    else:
                        action = np.random.normal(loc=self.action_distribution[step, 0], scale=self.action_distribution[step, 1])

                    # Predict the next state
                    state_action = torch.cat((torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)))
                    cur_state = self.dynamics_model(state_action).detach().numpy()

                    paths[p, step] = np.array([cur_state, action])

            # Evaluate the paths
            final_positions = np.apply_along_axis(self.forward_kinematics, arr=paths[:, -1, 0], axis=1)[:, 2]
            scores = -np.linalg.norm(final_positions - self.goal_state, axis=1)
            top_paths = paths[np.argsort(scores)[:config.CEM_TOP_PATHS]]

            # Resample the distribution
            top_k_mean = np.mean(top_paths[:, :, 1], axis=0)
            top_k_std = np.std(top_paths[:, :, 1], axis=0)
            top_k_dist = np.stack([top_k_mean, top_k_std], axis=1)
            if self.action_distribution is None:
                self.action_distribution = top_k_dist
            else:
                self.action_distribution += config.CEM_DISTRIBUTION_UPDATE*(top_k_dist - self.action_distribution)

        # Use the best path
        self.path = top_paths[0]

        # Visualise the path
        for i in range(config.CEM_PATH_LENGTH):
            pred_state = self.path[i, 0]
            pred_pos = self.forward_kinematics(pred_state)
            colour = (255*((1-float(i)/config.CEM_PATH_LENGTH)), 255*(float(i)/config.CEM_PATH_LENGTH), 0)
            v1 = VisualisationLine(pred_pos[0][0], pred_pos[0][1], pred_pos[1][0], pred_pos[1][1], colour, 0.005)
            v2 = VisualisationLine(pred_pos[1][0], pred_pos[1][1], pred_pos[2][0], pred_pos[2][1], colour, 0.005)
            self.planning_visualisation_lines += [v1, v2]

# The VisualisationLine class enables us to store a line segment which will be drawn to the screen
class VisualisationLine:
    # Initialise a new visualisation (a new line)
    def __init__(self, x1, y1, x2, y2, colour, width):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.colour = colour
        self.width = width


class DynamicsModel(nn.Module):
    def __init__(self):
        super(DynamicsModel, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
