# a file that handles taking a set of images files or a video file and splitting them to uniform
# data types for handing by the TrainingData.py methods

import cv2

from PIL import Image

from os import listdir, mkdir
from os.path import isfile, join, isdir

import Settings


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


# takes the video file at the specified path and converts it into a series of .png images
# the images are stored in a folder in the same location as the videoPath, the folder is called (videoName + "_split")
# videoPath: the folder with the video file, will be relative to images
# videoName: the name of the video file, excluding file extention, must be .mov
def videoToImages(videoPath, videoName):
    # determine the folder path
    videoPath = "images/" + videoPath + ""
    splitPath = videoPath + videoName + "_split"

    # if the directory doesn't exist, make one
    if not isdir(splitPath):
        mkdir(splitPath)

    video = cv2.VideoCapture(videoPath + videoName + ".mov")

    success, img = video.read()
    cnt = 0
    while success:
        num = str(cnt)
        while len(num) < Settings.IMG_NUM_DIGITS:
            num = "0" + num

        frameName = splitPath + "/img" + num + ".png"
        cv2.imwrite(frameName, img)
        if Settings.IMG_PRINT_STATUS:
            print("saved frame: " + frameName)

        cnt += 1
        success, img = video.read()


# get a list of Pillow images from a video file
# videoPath: the folder with the video file, will be relative to images
# videoName: the name of the video file, excluding file extention, must be .mov
# returns: a list of all the images
def videoToPillowImages(videoPath, videoName):
    # determine the folder path
    videoPath = "images/" + videoPath

    video = cv2.VideoCapture(videoPath + videoName + ".mov")

    success, img = video.read()
    if not success:
        print("failed to load video file " + videoName + " at " + videoPath)

    cnt = 0
    images = []

    while success:
        images.append(Image.fromarray(img))
        if Settings.IMG_PRINT_STATUS:
            print("loaded image " + str(cnt) + " " + str(images[-1]))

        cnt += 1
        success, img = video.read()

    return images
