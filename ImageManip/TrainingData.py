from PIL import Image
from os import listdir, mkdir
from os.path import isfile, join, isdir

import Settings


# take all the images in a folder and generate training data for all images
# the data is split into two folders, one is the gray scale input, in a folder called grayInput
# the other is the colored expected output in a folder called colorOutput
# all images will be the same aspect ratio as the originals, adding black bars to the sides or top
# if extra space needs to be filled
# all images will be the same number of pixels in width and height
# width: the number of pixels in the width of the output images, must be a positive integer > 0
# height: the number of pixels in the height of the output images, must be a positive integer > 0
# source: the path to the folder containing all images. The folder must only contain images.
#   The folder is relative to the images directory
#   or
#   a list of Pillow images to convert
# folder: the folder to save the images to, relative to images
def createTrainingData(width, height, source, folder):
    # direct the folder to the images folder
    folder = "images/" + folder + "/"

    # create the paths for the folders
    colorPath = folder + "colorOutput/"
    grayPath = folder + "grayInput/"

    # if the folders do not already exist, create them
    if not isdir(colorPath):
        mkdir(colorPath)

    if not isdir(grayPath):
        mkdir(grayPath)

    # get all the files in the given folder
    if isinstance(source, list):
        images = source
    else:
        images = []
        files = [file for file in listdir(source) if isfile(join(source, file))]

        # for each image, create the color image, then save it and convert it to gray and save it as well
        for f in files:
            # create the color image
            img = Image.open(source + f)
            images.append(img)

    # process all images into the training data and save them
    cnt = 0
    for i in images:
        num = str(cnt)
        while len(num) < Settings.IMG_NUM_DIGITS:
            num = "0" + num

        # create the color image
        i = scaleImage(width, height, i)
        # save the color image
        i.save(colorPath + "color" + num + ".png", "PNG")
        if Settings.IMG_PRINT_STATUS:
            print("saved color image: " + str(cnt) + " " + str(i))

        # create the gray image
        i = convertGrayScale(i)
        # save the gray image
        i.save(grayPath + "gray" + num + ".png", "PNG")
        if Settings.IMG_PRINT_STATUS:
            print("saved gray image: " + str(cnt) + " " + str(i))

        cnt += 1


# take the given PIL image and convert it to be resized to the given width and height, adding black
# bars to the sides or top and bottom if necessary to keep the same aspect ratio
# width: the number of pixels to resize the width of img to, must be a positive integer > 0
# height: the number of pixels to resize the height of img to, must be a positive integer > 0
# img: the image to convert, must be in RBG mode
def scaleImage(width, height, img):

    # create a black background
    background = Image.new("RGB", (width, height), color=(0, 0, 0))

    # set the alpha channels of both images
    background.putalpha(255)
    img.putalpha(255)

    # get the size of the given image
    imgW, imgH = img.size

    # find the ratios of the width and height to determine where black bars go
    wRatio = imgW / width
    hRatio = imgH / height

    # if the width is bigger, the black bars go on the top and bottom, otherwise the sides
    bigWidth = wRatio > hRatio
    if bigWidth:
        # the new width is the same as the desired width
        newW = width
        # the new height is based on the desired width and the ratio of the original image
        newH = int(round(width * imgH / imgW))
        space = int(round((height - newH) * .5))
    else:
        # the new width is based on the desired height and the ratio of the original image
        newW = int(round(height * imgW / imgH))
        # the new height is the same as the desired height
        newH = height
        space = int(round((width - newW) * .5))

    # resize the image to the desired width and height, maintaining the original aspect ratio
    img = img.resize((newW, newH), Image.CUBIC)

    # determine the bounds of where the resized image should be pasted based on black par positions
    if bigWidth:
        # black bars on the top and bottom
        bounds = (0, space, width, height - space)
    else:
        # black bars on the sides
        bounds = (space, 0, width - space, height)

    # paste the image
    background.paste(img, bounds)

    # return the resized image with black bars
    return background


# convert the given PIL image to gray scale and returns it
# img: the image to convert, must be in RBG mode
def convertGrayScale(img):
    # TODO a note, may want to find a way to avoid using putpixel() if performance suffers

    # get the size of the image
    width, height = img.size

    # for each pixel, turn it to gray scales
    for i in range(width):
        for j in range(height):
            # get the pixel value
            pix = img.getpixel((i, j))
            # calculate the gray scale value
            gray = int(.3 * pix[0] + .59 * pix[1] + .11 * pix[2])
            # set the pixel value to the gray scale value
            img.putpixel((i, j), (gray, gray, gray))

    return img
