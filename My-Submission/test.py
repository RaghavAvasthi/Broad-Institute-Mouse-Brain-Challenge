# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:25:03 2019

@author: Raghav Avasthi
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

import os

import tensorflow as tf
from tensorflow import keras



def crop_n_resize(image, border_margin = 10, size_x = 512, size_y = 512):
    '''
    Description
    -----------
    Given an image in 'uint8' format, the function does the following operations on the image to standardize it.
    > Finds the largest bounding box in the image containing the whole MRI object in the image.
    > Crops the image as per the bounding box leaving a margin specified by 'border_margin'
    > Pads the image to make it square shape. Padding is done by replicating the border pixels
    > Resizes the image to the desired size
    
    Parameters
    ----------
        image:
            Accepts a 3-channel image in uint8 format.
            
        border_margin:
            Accepts an integer depicting the amount of margin to be kept while cropping the image from its smallest bounding box possible. 
            Default is 10.
        
        size_x:
            Accepts an interger value. It is the desired width of the resultant image after resize. Default is 512.
        
        size_y:
            Accepts an interger value. It is the desired height of the resultant image after resize. Default is 512.
        
    Returns
    -------
        lat_concat:
            Returns a 3 channel 'uint8' numpy matrix stitched image made up of all the segments in a given magnification level.
            Returns None if inputs to the function do not pass the initial checks.
    '''
    if not isinstance(border_margin, int):
        print('\n' + 'ERROR: Border Margin supplied is not an integer. Stopping code execution' + '\n')
        sys.exit()
    if not isinstance(size_x, int):
        print('\n' + 'ERROR: Desired width of the output image supplied is not an integer. Stopping code execution' + '\n')
        sys.exit()
    if not isinstance(size_y, int):
        print('\n' + 'ERROR: Desired height of the output image supplied is not an integer. Stopping code execution' + '\n')
        sys.exit()
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # converting the image into grayscale
    _,bw = cv2.threshold(gray,210,255,cv2.THRESH_BINARY_INV) # converting graysacle image into binary for morphological operations
    bw = cv2.erode(bw, np.ones((5,5), np.uint8), 10) # Erosion and dilution to remove any salt and pepper noise
    bw = cv2.dilate(bw, np.ones((5,5), np.uint8), 10)
    label_bw = ski.measure.label(bw)
    del bw
    box_concat = None
    for region in ski.measure.regionprops(label_bw):
        box = np.asarray(region.bbox).reshape(4,1) # Calcualting bounding box for all objects in the image
        if box_concat is None: box_concat = box
        box_concat = np.concatenate((box_concat, box), axis = 1) # Concatinating all bounding box results to find the extrema points
    del label_bw
    (gray_row, gray_col) = gray.shape
    row_min = max(0, min(box_concat[0]) - border_margin)
    col_min = max(0, min(box_concat[1]) - border_margin) # Calcualted the extrema points
    row_max = min(gray_row, max(box_concat[2]) + border_margin)
    col_max = min(gray_col, max(box_concat[3]) + border_margin)
    del box_concat
    
    crop_im = image[row_min:row_max, col_min:col_max,:] # Cropped the image as per calculated extremas
#    crop_im = cv2.equalizeHist(crop_im) ###### ++++++++++++++++++
    (r,c,_) = crop_im.shape
    
    if r > c: # Made the image in square shape 
        diff = r-c
        pad = int(diff / 2)
        crop_im = cv2.copyMakeBorder(crop_im,0,0,pad,pad,cv2.BORDER_REPLICATE)# Used padding by replicating the border pixels to make square 
    elif r < c:
        diff = c-r
        pad = int(diff / 2)
        crop_im = cv2.copyMakeBorder(crop_im,pad,pad,0,0,cv2.BORDER_REPLICATE)
    crop_im = cv2.resize(crop_im,(size_x,size_y)) # resized the resultant image to the deisred size
    return crop_im


if __name__ == '__main__':
    model = keras.models.load_model(r'C:\Users\Raghav Avasthi\Desktop\mouse brain\Jupyter code\mouse_brain_model.h5')
    test_dir = r'C:\Users\Raghav Avasthi\Desktop\mouse brain\Brain_Dataset_Color\validate\sagittal'
    list_mri = os.listdir(test_dir)
    index=0
    for image_name in list_mri:
        image = np.asarray(cv2.imread(os.path.join(test_dir,image_name)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        final_im = crop_n_resize(image, size_x = 224, size_y = 224)
        check = np.full((1,224,224,3), 0) 
        check[0,:,:,:] = final_im
        prediction = model.predict(check)
        
        print('%n' + ' Prediction is = ' + str(prediction) + '%n')
        
        
