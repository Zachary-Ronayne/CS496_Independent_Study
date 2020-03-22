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
NET_PROPAGATION_RATE = .1

# The activation function used for backpropagation
COST_FUNC = "entropy"

# Constant to use for regularization, 0 to not use it
REGULARIZATION_CONSTANT = 0.00000001 * NET_PROPAGATION_RATE

# The activation function to use
ACTIVATION_FUNC = SIGMOID


# Image manipulation settings

# keeps track of if the reading and writing status of images should be printed to the console
IMG_PRINT_STATUS = True

# the number of digits in file outputs for frames
IMG_NUM_DIGITS = 8


# the target to use for CUDA
VECTORIZE_TARGET = "cpu"
