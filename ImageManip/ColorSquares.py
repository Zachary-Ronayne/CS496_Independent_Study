# a file that generates images for training data where each image is a square with different colors in a grid of squares

import random

from ImageManip.TrainingData import *


def getSquareImage(size, width, height):
    """
    Get one Pillow image of the ColorSquare form
    :param size: the size of each square
    :param width: the number of squares in the width
    :param height: the number of squares in the height
    :return: the image
    """
    # TODO add more options, specifically noise
    img = Image.new("RGB", (width * size, height * size), color=(0, 0, 0))

    for x in range(width):
        for y in range(height):
            color = int(random.uniform(0, 3))
            if color == 0:
                color = (255, 0, 0)
            elif color == 1:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            for i in range(size):
                for j in range(size):
                    img.putpixel((x * size + i, y * size + j), color)

    return img


def getSquareImages(number, size=1, width=1, height=1):
    """
    Get a  number of Pillow image of the ColorSquare form
    :param number: the number of images to get
    :param size: the size of each square in each ColorSquare
    :param width: the number of squares in the width of each image
    :param height: the number of squares in the height of each image
    :return: the images
    """
    return [getSquareImage(size, width, height) for i in range(number)]


def saveSquareTrainingData(name, location, number, size=1, width=1, height=1):
    """
    Generate training data based on color images.
    :param name: The name of the generated folder
    :param location: The location, relative to images, that the folder will be generated
    :param number: The number of images to generate
    :param size: The size of each square in each ColorSquare
    :param width: The number of squares in the width of each image
    :param height: The number of squares in the height of each image
    """
    imgs = getSquareImages(number=number, size=size, width=width, height=height)
    path = location + name + " " + "(train_data)"
    if not isdir("images/" + path):
        mkdir("images/" + path)

    folderToInOutImages(size * width, size * height, imgs, path)
