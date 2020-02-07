# A class for handling turning images into 1 dimensional lists for use with a feed forward neural network
# the lists act as input and output

from PIL import Image
from ImageManip import TrainingData


# take a PIL image, and return a tuple of input and output data for a Network
# the first element is the input data, the second element is the expect output
# img: the PIL image to process
# width: the desired with of the resized version of the image
# height: the desired height of the resized version of the image
def imageToData(img, width, height):
    # create the reszied color image
    color = TrainingData.scaleImage(width, height, img)
    # add all the color values to the expected output data
    exData = []
    pixels = color.load()
    for x in range(width):
        for y in range(height):
            p = pixels[x, y]
            exData.append(p[0] / 256.0)
            exData.append(p[1] / 256.0)
            exData.append(p[2] / 256.0)

    # create the gray scale version of the color image
    gray = TrainingData.convertGrayScale(color)
    # add all the color values to the input data
    inData = []
    pixels = gray.load()
    for x in range(width):
        for y in range(height):
            inData.append(pixels[x, y][0] / 256.0)

    # return the input data
    return inData, exData


# get a PIL image from the given list of data
# data: a 1 dimensional list of all the RGB values, first element is red, then green, then blue, then next pixel
#   the data should be in range [0, 1], which will be converted to [0, 256]
# width: the number of pixels in a row
# height: the number of pixels in a column
def dataToImage(data, width, height):
    img = Image.new("RGB", (width, height))

    cnt = 0
    for x in range(width):
        for y in range(height):
            img.putpixel((x, y),
                         (int(round(data[cnt] * 256)),
                          int(round(data[cnt + 1] * 256)),
                          int(round(data[cnt + 2] * 256)),
                          256))

            cnt += 3

    return img
