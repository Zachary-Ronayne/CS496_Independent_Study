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
NET_PROPAGATION_RATE = .01

# The cost function used for backpropagation
COST_FUNC = "entropy"

# Constant to use for regularization, 0 to disable
REGULARIZATION_CONSTANT = 0.0003 * NET_PROPAGATION_RATE

# Amount to decrease weights when trained with a backpropagation, set to 0 to disable
WEIGHT_SHRINK = 0.000001

# The activation function to use
ACTIVATION_FUNC = TANH

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
