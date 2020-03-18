# a file containing values for general settings for the code


# Neural network settings

# the absolute value of the maximum value a weight can take on
NET_MAX_WEIGHT = 2

# the absolute value of the maximum value a bias can take on
NET_MAX_BIAS = 2

# the rate at which changes are made in the backpropagation algorithm, larger values mean larger steps
#   meaning less precise training, but faster training
NET_PROPAGATION_RATE = 0.005

# The activation function used for backpropagation
ACTIVATION_FUNC = "entropy"

# Constant to use for regularization, 0 to not use it
REGULARIZATION_CONSTANT = 0.01 * NET_PROPAGATION_RATE


# Image manipulation settings

# keeps track of if the reading and writing status of images should be printed to the console
IMG_PRINT_STATUS = True

# the number of digits in file outputs for frames
IMG_NUM_DIGITS = 8
