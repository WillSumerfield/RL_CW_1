####################################
#      YOU MAY EDIT THIS FILE      #
# MOST OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn

# Imports from this project
import constants
import config

# Configure matplotlib for interactive mode
plt.ion()


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self, forward_kinematics):
        # Get the forward kinematics function from the environment
        self.forward_kinematics = forward_kinematics
        # A list of visualisations which will be displayed in the middle (planning) and right (policy) of the window
        self.planning_visualisation_lines = []
        self.policy_visualisation_lines = []
        self.trained = False
        self.step = 0

        self.model = ActionModel()
        self.path = []

        self.loss = np.zeros((config.MAX_DEMOS, config.ATTEMPTS))

        self.buffer = ReplayBuffer()
        self.demos = 0
        self.attempt = 0

    # Reset the robot at the start of an episode
    def reset(self):
        self.planning_visualisation_lines = []
        self.policy_visualisation_lines = []
        self.step = 0
        self.model = ActionModel()
        self.demos += 1

        if self.demos == config.MAX_DEMOS:
            self.demos = 0
            self.attempt += 1
            self.buffer = ReplayBuffer()
            if self.attempt == config.ATTEMPTS:
                plt.plot(list(range(1, config.MAX_DEMOS+1)), self.loss.mean(axis=1))
                plt.title("Goal Distance vs Num. Of Demos")
                plt.xlabel("Number of Demos")
                plt.ylabel("Goal Distance")
                plt.xticks(range(1, config.MAX_DEMOS+1))
                plt.yscale('log')
                plt.show()
                plt.savefig("Imitation.png")
                return False
            
        return True

    # Get the demonstrations
    def get_demos(self, demonstrator):
    
        # Train the model until convergence
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()
        
        # Add a demo
        print(f"Attempt: {self.attempt+1}, Demos: {self.demos+1}")
        for state, action in demonstrator.generate_demonstration():
            self.buffer.add_data(state, action)
    
        # Train a number of epochs
        for epoch in range(config.EPOCH_COUNT):

            # Get a random batch of data
            minibatches = self.buffer.sample_epoch_minibatches(config.BATCH_SIZE)
            for states, actions in minibatches:
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                predicted_actions = self.model(states)
                predicted_actions = torch.clip(predicted_actions, min=-constants.MAX_ACTION_MAGNITUDE, max=constants.MAX_ACTION_MAGNITUDE)
                # Compute the loss
                loss = criterion(predicted_actions, actions)
                # Backward pass
                loss.backward()
                # Optimize
                optimizer.step()

            #print(f"Epoch: {epoch}, Loss: {loss.item()}")

        self.model.eval()

    # Get the next action
    def select_action(self, state):
        action = self.model(torch.tensor(state, dtype=torch.float32)).detach().numpy()
        self.step += 1
        episode_done = self.step >= int(constants.CEM_PATH_LENGTH)

        if episode_done:
            self.loss[self.demos, self.attempt] = np.linalg.norm(constants.GOAL_STATE - self.forward_kinematics(state)[2])

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


class ActionModel(nn.Module):
    def __init__(self):
        super(ActionModel, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ReplayBuffer class stores transitions
class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.size = 0

    def add_data(self, state, action):
        self.states.append(state)
        self.actions.append(action)
        self.size += 1

    # Create minibatches for a single epoch of training (one epoch means all the training data is seen once)
    def sample_epoch_minibatches(self, minibatch_size):
        # Convert lists to NumPy arrays for indexing
        states_array = np.array(self.states)
        actions_array = np.array(self.actions)
        # Shuffle indices
        indices = np.random.permutation(self.size)
        minibatches = []
        # Create minibatches
        for i in range(0, self.size, minibatch_size):
            # Get the indices for this minibatch
            minibatch_indices = indices[i: i + minibatch_size]
            minibatch_states = states_array[minibatch_indices]
            minibatch_actions = actions_array[minibatch_indices]
            # Convert to torch tensors
            inputs = torch.tensor(minibatch_states, dtype=torch.float32)
            targets = torch.tensor(minibatch_actions, dtype=torch.float32)
            minibatches.append((inputs, targets))
        return minibatches