#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 09:54:47 2018

@author: daniel

File to create a comprehensive data set drawing from the original microscope
images, annotations generated by the tierpsy tracker, and 
hand-drawn annotations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utility_functions import load_image_into_numpy_array


# define the various directories
DATA_DIR = '/Users/daniel/Documents/UCL/Project/Data/'

#OUTPUT_DIR = os.path.join(DATA_DIR, 'collated_dataset')


# get list of file names
dirNames = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('synth')])
print("Filenames: ")
print("\n".join(dirNames))
print("")


# Loop through each of the datasets
for directory in dirNames:

    # set which dataset we are using
    print("Set being created: {}".format(directory))
    
    # define the directories of the images and outputs
    IMAGES_DIR = os.path.join(DATA_DIR, directory)
    CROPPED_OUTPUT_DIR = os.path.join(DATA_DIR, 'annotation-data', 'cropped_collated_dataset', directory)
    print(CROPPED_OUTPUT_DIR)
    
    FNAMES = [f for f in os.listdir(IMAGES_DIR) if not f.startswith('.')]
    FNAMES = sorted(FNAMES, key=int)
     

    for fname in FNAMES:
        print(fname)
        TEST_IMAGE_PATH = os.path.join(IMAGES_DIR, '{}/image/image_{}.png'.format(fname,fname))
        image = Image.open(TEST_IMAGE_PATH)
        
        img = load_image_into_numpy_array(image)

        MASKS_DIR = os.path.join(IMAGES_DIR, '{}/masks/'.format(fname))
        MASK_FNAMES = sorted([f for f in os.listdir(MASKS_DIR) if not f.startswith('.')])
    
    
        masks = []
        for m in MASK_FNAMES:
            mask = Image.open(os.path.join(MASKS_DIR,m))
            masks.append(np.asarray(mask)[:,:,:3])
        
        h = img.shape[0]
        w = img.shape[1]
        
        # Splitting the images into 16 smaller chunks
        # Originially the images are 2048*2048 however this is far too large
        # to fit in to GPU RAM. 
        for x in range(4):
            for y in range(4):
                
                # We loop through all the chunks and set pos_example = True if
                # the chunk contains at least part of a worm, keeping only
                # those chunks.
                pos_example = False
        
        
                j = 0
                
                # Loop through all of the worms 
                for m in masks: 
                    # crop the mask for the chunk being examined
                    cropped_mask = m[int((h/4)*x):int((h/4)*(x+1)), int((w/4)*y):int((w/4)*(y+1))]
        
                    # if the chunk contains a worm:
                    if np.any(cropped_mask):
                        # Create a subdir for the masks for this crop
                        os.makedirs(CROPPED_OUTPUT_DIR + '/{}_{}{}/masks'.format(fname,x,y), exist_ok=True)
                        mask_filename = CROPPED_OUTPUT_DIR + '/{}_{}{}/masks/mask_{}.png'.format(fname,x,y,j)
                        plt.imsave(fname=mask_filename, arr=cropped_mask, format='png', cmap='gray')
                        
                        # Set this flag True to save this image crop
                        pos_example = True
        
                    j += 1
        
                if pos_example:
                     os.makedirs(CROPPED_OUTPUT_DIR + '/{}_{}{}/image'.format(fname,x,y), exist_ok=True)
                     cropped_img = img[int((h/4)*x):int((h/4)*(x+1)), int((w/4)*y):int((w/4)*(y+1))]
                     image_filename = CROPPED_OUTPUT_DIR + '/{}_{}{}/image/image_{}_{}{}.png'.format(fname,x,y,fname,x,y)

                     plt.imsave(fname=image_filename, arr=cropped_img, format='png', cmap='gray')
    

        
#TODO: make this script output tf.record files (sharded by set?)
        
        
        
