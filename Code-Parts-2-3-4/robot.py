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
from collections import deque

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
        self.dynamics_model_data = deque(maxlen=config.DYNAMICS_DATA_COLLECTION_EPISODE_LENGTH*config.DYNAMICS_DATA_COLLECTION_EPISODES)
        # The losses to graph
        self.closed_loss = []
        self.open_loss = []
        # The dynamics model
        self.dynamics_model = DynamicsModel()
        # Track if the dynamics model is initialized
        self.dynamics_model_init = False
        # The CEM distribution
        self.action_distribution = None
        # The current closed-loop path
        self.path = None
        # The number of episodes that the model has been trained on
        self.trained_episodes = 0
        self.open_mode = True

        # Make plan, then reset. All in one graph

    # Reset the robot at the start of an episode
    def reset(self):
        self.episode_steps = 0
        self.model_visualisation_lines = []
        self.planning_visualisation_lines = []
        self.path = None

        # Plot the loss over time if we reached the end of training
        if self.dynamics_episode_count == config.DYNAMICS_DATA_COLLECTION_EPISODES:
            print(f"Episode: {self.trained_episodes}")
            if self.trained_episodes == config.TOTAL_EPISODES:
                if self.open_mode:
                    config.FRAME_RATE = 100
                    self.action_distribution
                    self.trained_episodes = 0
                    self.open_mode = False
                    self.trained_episodes = -1
                    self.dynamics_episode_count = 0
                    self.dynamics_model_init = False
                    self.dynamics_model = DynamicsModel()
                    self.dynamics_model_data = deque(maxlen=config.DYNAMICS_DATA_COLLECTION_EPISODE_LENGTH*config.DYNAMICS_DATA_COLLECTION_EPISODES)
                else:
                    plt.plot(self.closed_loss, label="Closed-Loop Loss")
                    plt.plot(self.open_loss, label="Open-Loop Loss")
                    plt.title("Dynamics Model Loss")
                    plt.xlabel("Episode Number")
                    plt.ylabel("Loss")
                    plt.xticks(range(0, config.TOTAL_EPISODES+1))
                    #plt.yscale('log')
                    plt.legend()
                    plt.show()
                    plt.savefig("Mix_Loss.png")
                    return True
            self.trained_episodes += 1
        return False

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
            if not self.dynamics_model_init:
                self.train_dynamic_model()
                self.dynamics_model_init = True

            # Use CEM to make a plan each episode
            if self.path is None or not self.open_mode:
                self.create_plan(state)

            # Train the dynamics model for a set number of minibatches
            self.train_dynamic_model(batches=config.DYNAMICS_BATCH_SIZE)

            pred_state = self.path[self.episode_steps if self.open_mode else 0, 0]
            action = self.path[self.episode_steps if self.open_mode else 0, 1]

            if self.episode_steps == config.TOTAL_EPISODES:
                loss = np.linalg.norm(constants.GOAL_STATE - self.forward_kinematics(state)[2])
                if self.open_mode:
                    self.open_loss += [loss]
                else:
                    self.closed_loss += [loss]

            # Visualise the action
            pred_pos = self.forward_kinematics(pred_state)
            state_pos = self.forward_kinematics(state)
            v1 = VisualisationLine(state_pos[2][0], state_pos[2][1], pred_pos[2][0], pred_pos[2][1], (0, 255, 0), 0.005)
            self.model_visualisation_lines = [v1]

        self.episode_steps += 1
        episode_done = self.episode_steps >= config.CEM_PATH_LENGTH

        return action, episode_done

    def train_dynamic_model(self, batches=None):
        self.dynamics_model.train()
        optimizer = optim.Adam(self.dynamics_model.parameters())
        criterion = nn.MSELoss()

        # Train until the loss is below a threshold
        avg_loss = 0
        batch_ind = 0
        while True:
            # Get a random batch of data
            batch = np.array([self.dynamics_model_data[i] for i in np.random.choice(len(self.dynamics_model_data), config.DYNAMICS_BATCH_SIZE)])
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

            avg_loss += loss.item()

            batch_ind += 1
            #if batch_ind % 50 == 0:
                #print(f"Batches: {batch_ind}, Loss: {loss.item()}")

            if not batches is None:
                if batch_ind == batches:
                    break
            elif loss.item() < config.DYNAMICS_TRAINING_THRESHOLD:
                break

        self.dynamics_model.eval()
        
        #self.dynamics_model_loss += [avg_loss/batch_ind]

    # Function to add a transition to the buffer
    def add_transition(self, state, action, next_state):
        self.dynamics_model_data.append(np.array([state, action, next_state]))

    # Create a new path plan using our dymamics model
    def create_plan(self, state):

        # Iteratively improve the plans
        action_distribution = None
        for itr in range(config.CEM_ITERATIONS):

            # Get a series of plans
            paths =  np.zeros((config.CEM_PATHS, config.CEM_PATH_LENGTH-self.episode_steps, 2, 2))
            for p in range(config.CEM_PATHS):
                cur_state = state

                # Generate a path
                for step in range(config.CEM_PATH_LENGTH-self.episode_steps):

                    # If this is our first pass, use a normal distribution
                    if action_distribution is None:
                        action = np.random.uniform(low=-constants.MAX_ACTION_MAGNITUDE, high=constants.MAX_ACTION_MAGNITUDE, size=2)
                    else:
                        action = np.clip(np.random.normal(loc=action_distribution[step, 0], scale=action_distribution[step, 1]),
                                         -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)

                    # Predict the next state
                    state_action = torch.cat((torch.tensor(cur_state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)))
                    cur_state = self.dynamics_model(state_action).detach().numpy()

                    paths[p, step] = np.array([cur_state, action])

            # Evaluate the paths
            final_positions = np.apply_along_axis(self.forward_kinematics, arr=paths[:, -1, 0], axis=1)[:, 2]
            scores = np.linalg.norm(final_positions - self.goal_state, axis=1)
            top_paths = paths[np.argsort(scores)[:config.CEM_TOP_PATHS]]

            # Resample the distribution
            top_k_mean = np.mean(top_paths[:, :, 1], axis=0)
            top_k_std = np.std(top_paths[:, :, 1], axis=0)
            top_k_dist = np.stack([top_k_mean, top_k_std], axis=1)
            if self.action_distribution is None:
                action_distribution = top_k_dist
            else:
                action_distribution += config.CEM_DISTRIBUTION_UPDATE*(top_k_dist - self.action_distribution)

        self.path = top_paths[0]

        # Visualise the paths
        self.planning_visualisation_lines = []
        for i in range(config.CEM_PATH_LENGTH - self.episode_steps):
            pred_state = self.path[i, 0]
            pred_pos = self.forward_kinematics(pred_state)
            colour = (255*(1 - float(i)/config.CEM_PATH_LENGTH), 255*(float(i)/config.CEM_PATH_LENGTH), 0)
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
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
