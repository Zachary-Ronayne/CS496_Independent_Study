from NeuralNet.FeedForward import *

from ImageManip.TrainingData import *


class ImageNet(MatrixNetwork):
    """
    A class that handles a MatrixNetwork that processes images, providing convenience methods.
    """

    def __init__(self, inSize, outSize, inner):
        """
        Initialize a network for use with converting gray scale images to color images
        :param inSize: A tuple of (width, height) for the size of input images
        :param outSize: A tuple of (width, height) for the size of output images
        :param inner: The number of nodes in each inner layer in the form, [hidden 1, hidden 2, ..., hidden n]
        """
        self.inSize = inSize
        self.outSize = outSize

        # get the network of appropriate size
        net = makeImageNetwork(inSize, outSize, inner, matrixNet=True)
        self.sizes = net.sizes

        # call super to initialize the rest of this MatrixNetwork correctly
        super().__init__(self.sizes)

    def processImage(self, img):
        """
        Take a PIL image, should be gray scale, and process it into a color image, using this network.
        :param img: The image to process
        :return: The image resulting from this network
        """

        # ensure that the image is of the correct dimensions
        img = scaleImage(self.inSize[0], self.inSize[1], img)
        # get the data from the image for the Network
        data = grayImageToData(img)
        # determine the outputs of the Network for the final image
        outs = self.calculateInputs(data)
        # turn the Network outputs into an image
        return dataToImage(outs, self.outSize[0], self.outSize[1])

    def save(self, name):
        # save data from the MatrixNetwork
        super().save(name)
        # save the image input and output sizes
        with open("saves/" + name + ".txt", "a") as f:
            f.write(str(self.inSize[0]) + " " + str(self.inSize[1]) + "\n")
            f.write(str(self.outSize[0]) + " " + str(self.outSize[1]) + "\n")

    def load(self, name):
        # load data fom the MatrixNetwork
        super().load(name)
        # load the data again, but this time loading the last two lines for the in and output sizes
        # TODO This is inefficient, should rework to only have to load once
        with open("saves/" + name + ".txt", "r") as f:
            lines = f.readlines()[-2:]
            self.inSize = tuple(lines[0])
            self.outSize = tuple(lines[1])
