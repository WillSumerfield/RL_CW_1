##########################
# YOU MAY EDIT THIS FILE #
##########################

# The window width and height in pixels, for both the "environment" window and the "planning" window.
# If you wish, you can modify this according to the size of your screen.
WINDOW_SIZE = 600

# The frame rate for pygame, which determines how quickly the program runs.
# Specifically, this is the number of time steps per second that the robot will execute an action in the environment.
# You may wish to slow this down to observe the robot's movement, or speed it up to run large-scale experiments.
FRAME_RATE = 10

# You may want to add your own configuration variables here, depending on the algorithm you implement.
DYNAMICS_DATA_COLLECTION_EPISODES = 100
DYNAMICS_DATA_COLLECTION_EPISODE_LENGTH = 5
DYNAMICS_BATCH_SIZE = 128
DYNAMICS_TRAINING_THRESHOLD = 0.0001
DYNAMICS_BATCH_COUNT = 1000

CEM_ITERATIONS = 10
CEM_PATHS = 50
CEM_PATH_LENGTH = 30
CEM_TOP_PATHS = 10
CEM_DISTRIBUTION_UPDATE = 0.5

TOTAL_EPISODES = 20