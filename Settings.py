# a file containing values for general settings for the code


# constants for FeedForward

SIGMOID = 0
TANH = 1
RELU = 2

# Neural network settings

# the absolute value of the maximum value a weight can take on
NET_MAX_WEIGHT = 2

# the absolute value of the maximum value a bias can take on
NET_MAX_BIAS = 2

# the rate at which changes are made in the backpropagation algorithm, larger values mean larger steps
#   meaning less precise training, but faster training
NET_PROPAGATION_RATE = 0.1

# Constant to use for regularization, 0 to disable
REGULARIZATION_CONSTANT = 0.01

# Amount to decrease weights when trained with backpropagation, set to 0 to disable
WEIGHT_SHRINK = 0.00001

# The amount of dropout that should be used. Use None to disable dropout.
# Otherwise, use a number in the range [0, 1] for that percentage of the nodes to be dropped out,
#   meaning, 0.1 gives a 10% chance for each node to be dropped out,
#   .9 gives a 90% chance for each node to be dropped out
DROP_OUT = 0.5

# The cost function used for backpropagation
COST_FUNC = "entropy"

# The activation function to use
ACTIVATION_FUNC = SIGMOID

# The way the learning rate changes as backpropagation gets towards the input layer.
# Use 0 to not change, a negative number to decrease learning rate as the input layer approaches,
#   and a positive number to increase learning rate as input later approaches.
# Recommended value is -1
LEARNING_RATE_BY_LAYER = -1


# Image manipulation settings

# keeps track of if the reading and writing status of images should be printed to the console
IMG_PRINT_STATUS = True

# the number of digits in file outputs for frames
IMG_NUM_DIGITS = 8


# the target to use for CUDA
VECTORIZE_TARGET = "cpu"
