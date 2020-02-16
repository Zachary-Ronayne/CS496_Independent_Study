# a file that handles taking a set of images files or a video file and splitting them to uniform
# data types for handing by the TrainingData.py methods

import cv2
import numpy as np

import Settings

from PIL import Image
from os import listdir, mkdir
from os.path import isfile, join, isdir


# load in a folder of images, then convert them all to .png images and store them in a new folder
# all files in folder must be images
# video: the path to the folder of images relative to the images folder of the program
def imagesToPng(imageLoc):
    directory = "images/" + imageLoc
    newDir = directory + "/split_" + imageLoc

    # load in all files
    files = [file for file in listdir(directory) if isfile(join(directory, file))]

    # create a directory for the new set of files
    if not isdir(newDir):
        mkdir(newDir)

    # save each of the .tif files as a .png file in a new folder
    cnt = 0
    for f in files:
        num = str(cnt)
        while len(num) < Settings.IMG_NUM_DIGITS:
            num = "0" + num

        file = directory + "/" + str(f)
        img = Image.open(file)
        newName = newDir + "/" + imageLoc + "_" + num + ".png"
        try:
            img.save(newName, "PNG")
            cnt += 1
        except OSError:
            if Settings.IMG_PRINT_STATUS:
                print("filed to save " + newName)
        else:
            if Settings.IMG_PRINT_STATUS:
                print("saved " + newName)


# take the video file at the specified path and convert it into a series of .png images
# the images are stored in a folder in the same location as the videoPath, the folder is called (videoName + "_split")
# returns a string representing a path to the split folder
# videoPath: the folder with the video file, will be relative to images
# videoName: the name of the video file, excluding file extension, must be .mov
# size: a tuple with the width and height to resize the images to, don't include to not modify the size
# skip: skip every this many frames, default 1, meaning skip no frames
# start: the percentage in range [0, 1] of the starting point in the video to produce images, must be < end
# end: the percentage in range [0, 1] of th end point in the video tp produce images, must be > start
#   can also use integers for start and end along with the flag frameRange set to true to use a range of frames for
#   start and end
# frameRange: True to make start and end act as frame ranges, False to make them act as percentage ranges
def splitVideo(videoPath, videoName, size=None, skip=1, start=0, end=1, frameRange=False):
    # determine the folder pat
    splitPath = "images/" + videoPath + videoName + "_split"

    # if the directory doesn't exist, make one
    if not isdir(splitPath):
        mkdir(splitPath)

    # get the image files
    images = videoToPillowImages(videoPath, videoName, size, skip, start, end, frameRange)

    # go through each image and save them
    for i, img in enumerate(images):
        # determine the number for the file name of the image
        num = str(i)
        while len(num) < Settings.IMG_NUM_DIGITS:
            num = "0" + num

        # determine the name of the file
        frameName = splitPath + "/img" + num + ".png"
        # write the file
        cv2.imwrite(frameName, np.array(img))
        # print that the frame was saved, if applicable
        if Settings.IMG_PRINT_STATUS:
            print("saved frame: " + frameName)

    return splitPath + "/"


# get a list of Pillow images from a video file
# videoPath: the folder with the video file, will be relative to images
# videoName: the name of the video file, excluding file extention, must be .mov
# returns: a list of all the images
# see videoToImages for extra parameter descriptions
def videoToPillowImages(videoPath, videoName, size=None, skip=1, start=0, end=1, frameRange=False):
    # determine the folder path
    videoPath = "images/" + videoPath
    filePath = videoPath + videoName + ".mov"

    # load in the video file
    video = cv2.VideoCapture(filePath)

    # load the first frame to ensure that the video loaded correctly
    success, img = video.read()
    if not success:
        print("failed to load video file " + videoName + " at " + videoPath)
        return None

    # set up variables for loop
    cnt = 1
    images = []

    # find the total number of frames
    frameTotal = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # continue until the video can no longer load a frame
    while success:
        # determine the position for comparison to see if this frame is in the range of the video
        if frameRange:
            pos = cnt
        else:
            pos = cnt / frameTotal
            # if this frame is in the range of the video, and shouldn't be skipped
        if cnt % skip == 0 and start <= pos <= end:
            # convert the image array to a PIL image
            # TODO sometimes need to change the color mode here? with the commented out line
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            # if resizing should be done, then do so
            if size is not None:
                img = scaleImage(size[0], size[1], img)

            # add the image to the list
            images.append(img)

            # print information about loading the image, if applicable
            if Settings.IMG_PRINT_STATUS:
                print("loaded image " + str(cnt) + " " + str(images[-1]))

        # if the end of the range has been reached, end the loop
        if pos > end:
            break

        # increment the frame counter and load the next frame
        cnt += 1
        success, img = video.read()

    return images


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
