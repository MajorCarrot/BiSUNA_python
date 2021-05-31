from enum import auto, Enum

PARAMETERS = True
CONTINUOUS_PARAM = True

if CONTINUOUS_PARAM:
    ParameterType = float
    MAXIMUM_WEIGHT = 2147483647.0
    WEIGHT_MUTATION_CHANGE_PERCENTAGE = 1.0
    # 10 = 1000% change, 1 = 100% change possible
    EXCITATION_THRESHOLD = 0.0
    # minimum excitation necessary to activate the neuron
    MAX_NEURON_OUTPUT = 1000000
    REMAINING_NEURON_THRESHOLD = 0.001
    HALF_WEIGHT = 0
    MID_WEIGHT = 0
else:
    ParameterType = int
    MAXIMUM_WEIGHT = 65535
    WEIGHT_MUTATION_CHANGE_PERCENTAGE = 1
    EXCITATION_THRESHOLD = 256
    # minimum excitation necessary to deactivate the neuron
    # This constant considers the number of bits an excitation neuron must have in order
    # to trigger its actions.
    EXCITATION_THRESHOLD_BITS = 4
    # EXCITATION_THRESHOLD_BITS 6
    HALF_WEIGHT = 255
    MID_WEIGHT = 32767
    MAX_NEURON_OUTPUT = 65535
    REMAINING_NEURON_THRESHOLD = 15

EXTREME_NEGATIVE_REWARD = -1000000
NUMBER_OF_INITIAL_MUTATIONS = 200
NUMBER_OF_STEP_MUTATIONS = 5
INITIAL_ALLOCATION_LENGTH = 100000
SUBPOPULATION_SIZE = 100

EPISODES_PER_INDIVIDUAL = 1.0


class subpopulations(Enum):
    FITNESS_BASED = auto()
    NEURON_EFFICIENT = auto()
    CONNECTION_EFFICIENT = auto()
    NEURON_RICH = auto()
    CONNECTION_RICH = auto()
    NUMBER_OF_SUBPOPULATIONS = auto()


MAIN_SUBPOP = subpopulations.FITNESS_BASED

# spectrum diversity
SPECTRUM_SIZE = 6
NOVELTY_MAP_SIZE = 20
NUMBER_OF_SUBPOPULATIONS = 1
# NORMALIZED_SPECTRUM_DIVERSITY
# 1-add neuron, 2-del neuron, 3-add connection, 4-del connection
# MUTATION_PROBABILITIES = {0.25, 0.05, 0.6, 0.1}
MUTATION_PROBABILITIES = [0.01, 0.01, 0.49, 0.49]
CHANCE_OF_NEUROMODULATION = 0.1
CHANCE_OF_CONTROL_NEURON = 0.2
CHANCE_OF_WEIGHT_MUTATION = 0.5

# MUTATION_PROBABILITIES = {0.55, 0.1, 0.20, 0.15}
# CHANCE_OF_NEUROMODULATION = 0.2
# CHANCE_OF_CONTROL_NEURON = 0.5
# CHANCE_OF_WEIGHT_MUTATION = 0.7

#   ----------  Environments  ---------- #

# general features

# SET_NORMALIZED_INPUT
# SET_NORMALIZED_OUTPUT
# stop and exit the process if the max steps are reached in one trial, it means that the algorithm reached the best solution
# TERMINATE_IF_MAX_STEPS_REACHED

# # # # Reinforcement Learning# # # /

# Mountain Car
MAX_MOUNTAIN_CAR_STEPS = 1000
# #define	NOISY_MOUNTAIN_CAR
CONTINUOUS_MOUNTAIN_CAR = True
# if defined only one continuous output, otherwise 3 outputs

# CHANGING_MCAR_MAX_VELOCITY
MODIFIED_MCAR_MAX_VELOCITY = 0.04
# MCAR_TRIALS_TO_CHANGE = 10000
# MCAR_TRIALS_TO_CHANGE = 100
MCAR_TRIALS_TO_CHANGE = 1

MCAR_MIN_POSITION = -1.2
MCAR_MAX_POSITION = 0.6
MCAR_MAX_VELOCITY = 0.07
# the negative of this in the minimum velocity
MCAR_GOAL_POSITION = 0.6

# Double Pole Balancing
MAX_DOUBLE_POLE_STEPS = 100000
NON_MARKOV_DOUBLE_POLE = True

# Function Approximation
SEQUENTIAL_FUNCTION_APPROXIMATION = True
# SUPERVISED_FUNCTION_APPROXIMATION = True
# NUMBER_OF_FUNCTION_APPROXIMATION = 2
# MULTIPLE_RANDOM_FUNCTION_APPROXIMATION = True

# Single Pole Balancing
# MAX_SINGLE_POLE_STEPS = 10000
MAX_SINGLE_POLE_STEPS = 10000
RANDOM_START = False
