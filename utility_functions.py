#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 21:16:10 2018

@author: daniel

Utility functions for creating the worms dataset
and training the tf mask-RCNN model

#image loading and inference functions from https://github.com/tensorflow/models/tree/master/research/object_detection

"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from copy import copy
import tensorflow as tf

import sys
sys.path.append("..")
from object_detection.utils import ops as utils_ops

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, -1)).astype(np.uint8)[:,:,:3]

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

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

