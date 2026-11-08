# All hyperparameters and settings in one place

# Game
GAME_URL = "https://elvisyjlin.github.io/t-rex-runner/"

# Image state
# Canvas is 600×150 (4:1 ratio). We resize to 80×20 to preserve the aspect
# ratio — squashing to 80×80 would distort obstacle shapes significantly.
# The CNN input channels are FRAME_STACK stacked frames: (4, 20, 80).
FRAME_WIDTH  = 80
FRAME_HEIGHT = 20
FRAME_STACK  = 4   # stack N frames so the model can see motion

# Actions: 0=nothing, 1=jump. Duck removed: only cacti appear at the fixed speed, and a key tap never triggered ducking.
N_ACTIONS = 2

# DQN
LEARNING_RATE = 1e-4
GAMMA         = 0.99

# Epsilon: start low (10%) and decay linearly over EPSILON_DECAY_STEPS
# to a near-zero floor.
EPSILON_START      = 0.1
EPSILON_END        = 0.0001
EPSILON_DECAY_STEPS = 100_000  # individual timesteps over which to anneal

BATCH_SIZE         = 16
REPLAY_BUFFER_SIZE = 50_000
TARGET_UPDATE_FREQ = 1_000    # every N *steps* (not episodes) for consistency

# Pure-observation phase: collect this many steps before any weight updates
OBSERVE_STEPS = 100

# Training
MAX_EPISODES         = 3000
MAX_STEPS_PER_EPISODE = 10_000
MAX_TOTAL_STEPS      = 150_000  # total environment steps in training (eval steps do not count)

# Paths
MODEL_SAVE_PATH = "best_model.pth"

# Game timing
GAME_TIME_PER_STEP = 0.05  # about 3 game frames per step (fixed game time)

# Reproducibility and evaluation
SEED = 42
EVAL_EVERY = 100  # evaluate every N episodes
EVAL_EPISODES = 3  # number of evaluation episodes
EVAL_MAX_STEPS = 1500  # max steps per eval episode (~2 minutes wall time)
