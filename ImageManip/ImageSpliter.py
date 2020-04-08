# A file that contains methods for splitting one PIL image into multiple sub images for training data

from PIL import Image
import math

from ImageManip.TrainingData import folderToInOutImages


def subImages(source, width, height, resize=None):
    """
    Take a given image and return a list of PIL images of sub images from the original image.
    Images in the list are organized by row then column.
    Any extra space that the final row and columns will have from width and height not dividing
        evenly into the source dimensions, will be filled with black pixels
    :param source: The image to split
    :param width: The width of sub images
    :param height: The height of sub images
    :param resize: A 2-tuple for the height and width to resize the source image to
    :return: The list of sub images
    """

    # resize the source image, if applicable
    if resize is not None:
        source = source.resize(resize)

    # get the source image dimensions
    imgW, imgH = source.size

    # find the number of sub images in each row and column
    rows = math.ceil(imgH / height)
    cols = math.ceil(imgW / width)

    # initialize list for images
    imgs = []

    # loop through each row
    for r in range(rows):
        # loop through each column
        for c in range(cols):
            # get the sub image and place it in the images list
            x = width * c
            y = height * r
            imgs.append(source.crop((x, y, x + width, y + height)))

    # return the completed list
    return imgs


def splitImage(imgPath, storePath, width, height, resize=None):
    """
    Take an image at the given path and, in the given path, store the image as training data.
    :param imgPath: The path to the image to split, relative to images
    :param storePath: The path to store the training data, relative to images
    :param width: The width of sub images
    :param height: The height of sub images
    :param resize: A 2-tuple of the dimensions to resize the image at the given path to before it is split.
        Default: None
    """
    img = Image.open("images/" + imgPath)
    imgs = subImages(img, width, height, resize=resize)
    folderToInOutImages((width, height), (width, height), imgs, storePath)


def applyNetwork(img, net, resize=None):
    """
    Take the given PIL image and apply the network representing a sub image over the full image.
    Think of img as the entire space, and net processes a single tile of that space at a time.
    It is assumed that img is a gray scale image where all RGB values are the same.
    :param img: The image to process
    :param net: The Network to process the image with, must be an ImageNet where inSize and outSize are equal
    :param resize: A 2-tuple of the dimensions to resize the image at the given path to before it is processed.
        Default: None
    :return: the processed image
    """

    # resize the image
    if resize is not None:
        img = img.resize(resize)

    # get image sizes
    w, h = net.outSize
    sourceSize = img.size
    slicesX = math.ceil(sourceSize[0] / w)
    slicesY = math.ceil(sourceSize[1] / h)

    # split the images to appropriate sizes
    split = subImages(img, w, h)
    # process each slice
    split = [net.processImage(i) for i in split]

    # rebuild the image
    img = Image.new("RGB", (w * slicesX, h * slicesY))
    for y in range(slicesY):
        for x in range(slicesX):
            img.paste(split[x + y * slicesX], (x * w, y * h))

    # return the final image
    return img.crop((0, 0, sourceSize[0], sourceSize[1]))
