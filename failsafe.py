"""
Functions providing a failsafe in case we fail to detect the optic disc in the regular way. 
We will then call the detection_by_mean function, which  
takes the mean position of the optical disc for the right or left eye and draw a circle based on said position.+
"""

import cv2
import pandas as pds
import matplotlib.pyplot as plt

# We load the stats we previously did (ground truth)
df = pds.read_excel('stats_diametre.ods', sheet_name=None)

"""
Draws a circle on the mean optic disc position of a given picture. 
If the picture is of a left eye, we take the left eye stats, and if it is a right eye picture, we take the right eye stats.
param : image_name (string) - the name of the current picture 

"""
def detection_by_mean(image_name):
    # Load the image 
    image = load_and_resize(image_name)

    # Choose which spreadsheet to load
    if "OD" in image_name:
        current_sheet = df["OEIL DROIT"]
    else: 
        current_sheet = df["OEIL GAUCHE"] 

    # get the mean values of said spreadsheet 
    pos_x, pos_y, diam = get_mean_values(current_sheet)

    # Draw on top of the image 
    cv2.circle(image,(int(pos_x), int(pos_y)), diam, (255,0,0), 5)
    return image




"""
Function loading an image and resizing it to the correct format, based on its name.
param : image_name (string) - the name of the current picture 
returns : the image, loaded via cv2 and resized
"""
def load_and_resize(image_name):
    # load the image 
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    # resize it
    img = cv2.resize(img, (874, 538))
    return img


"""
Function reading the current sheet (right eye or left eye) and returning  useful values to draw our circle
returns :
    pos_x : (float) the mean x position of the optic disc for this kind of eye 
    pos_y : (float) the mean y position of the optic disc for this kind of eye 
    diam : (int) : the mean diameter of the optic disc for this kind of eye
"""
def get_mean_values(current_sheet): 


    # We do stats on the last value (mean)
    pos_x = current_sheet.pos_centre_x[-1]
    pos_y = current_sheet.pos_centre_y[-1]

    diam_x = current_sheet.Diam_x_Rouge[-1]
    diam_y = current_sheet.Diam_y_Rouge[-1]
    # The diameter is the mean of the x diameter and the y diameter (approximate the ellipsis as a circle)
    diam = int((diam_x + diam_y)//2)
    return pos_x, pos_y, diam