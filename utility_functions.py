#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 21:16:10 2018

@author: daniel

Utility functions for creating the worms dataset
and training the tf mask-RCNN model
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from copy import copy

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)[:,:,:3]

def masks_from_XML(annotation_path, img, full_masks=True):
    '''
    Args:
        annotation_path: path to XML annotations
        full_masks: Whether to return full size binary masks, 
                    or just the outlines
    Returns:
        masks: list of masks
        heads: list of co-ordinates of worm heads
    '''
    
    
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    masks = []
    heads = []
    
    for worm in root:
    # worm = root[2]
        cnt1 = []
        cnt2 = []
        side1 = worm[0]
        side2 = worm[1]
    
        try:
            for point in side1:
                x = int(point[0].text)
                y = int(point[1].text)
                cnt1.append(copy([x,y]))
    
            for point in side2:
                x = int(point[0].text)
                y = int(point[1].text)
                cnt2.append(copy([x,y]))
    
        except:
            x1=[]
            for x in side1[0::2]:
                x1.append(x.text) 
            y1 = []
            for y in side1[1::2]:
                y1.append(y.text)
    
            cnt1 = np.column_stack((x1,y1))
    
    
            x2=[]
            for x in side2[0::2]:
                x2.append(x.text) 
            y2 = []
            for y in side2[1::2]:
                y2.append(y.text)
            cnt2 = np.column_stack((x2,y2))
    
    
        # Check if the worms have been annotated HT-HT or HT-TH, and arrange 
        # the lists accordingly
        if (abs(int(cnt1[-1][0]) - int(cnt2[0][0])) < 10) and (abs(int(cnt1[-1][1]) - int(cnt2[0][1])) < 10):
            cnt_close = np.vstack([cnt1, cnt2])
        else:
            cnt_close = np.vstack([cnt1, cnt2[-1::-1]])
    
        heads.append(cnt_close[0])
    
        mask = np.zeros((img.shape[0], img.shape[1]))
        
        if full_masks:
            cv2.fillPoly(mask, pts =[np.int32(cnt_close)], color=(255,255,255))
        else:
            # alternative function for plotting only the outline for review of annotations
            cv2.polylines(mask, pts =[np.int32(cnt_close)],isClosed=True, color=(255,255,255))
        
        masks.append(copy(mask))
        
    return (masks, heads)



