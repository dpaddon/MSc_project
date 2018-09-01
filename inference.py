import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")
from object_detection.utils import ops as utils_ops
if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from copy import copy
import cv2
from skimage.morphology import skeletonize
import time
import datetime
import pickle

# This is needed to display the images.
#%matplotlib inline

from tensorflow.models.research.object_detection.utils import label_map_util
from tensorflow.models.research.object_detection.utils import visualization_utils as vis_util

from utility_functions import load_image_into_numpy_array
from utility_functions import run_inference_for_single_image


# Load inference graph from checkpoint
MODEL_NAME = 'resnet_50_atrous/fine_tuned_model_100k_final_sets'
PATH_TO_CKPT = os.path.join('./downloaded_models', MODEL_NAME, 'frozen_inference_graph.pb')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
    
# Load label map from file
PATH_TO_LABELS = './data/worm_label_map.pbtxt'
NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



visualise_outputs = False
save_anns_to_file = False
save_overlays_to_file = True
IMAGE_SIZE = (40,40)
NUM_IMAGES = 10


##############################################################################################################
# Load the datasets for inference

DATASET_DIR = './data/fullsize_images/'
datasets = [f for f in os.listdir(DATASET_DIR) if not f.startswith('.')]
print(datasets)



for dataset in datasets:
    
    
    print()
    print("########################################################################################################################")
    print("Using model: {}".format(MODEL_NAME))
    print()
    print("Running inference on dataset {}".format(dataset))
    print()
    print("########################################################################################################################")
    
    
    PATH_TO_TEST_IMAGES_DIR = os.path.join(DATASET_DIR, dataset)
    
    
    FNAMES = [f for f in os.listdir(PATH_TO_TEST_IMAGES_DIR) if not f.startswith('.')]
    FNAMES = sorted(FNAMES, key=int)[:NUM_IMAGES]
    
    # Size of output images.
    
    output_dicts_list = []
    inference_times = []
    
    ########################################################################################################################
    # Perform actual inference for each image
    ########################################################################################################################
    
    for fName in FNAMES:
        
        image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}/image/image_{}.png'.format(fName,fName))
    
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        
        start = datetime.datetime.now()
    
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        
        end = datetime.datetime.now()
        elapsed = end - start
        print("Inference for one image: {}.{}s".format(elapsed.seconds,round(elapsed.microseconds,2))) 
        inference_times.append(elapsed.total_seconds())
    
        if visualise_outputs:
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              line_thickness=2)
            plt.figure(figsize=IMAGE_SIZE)
            plt.title("Image {}".format(image_path[:-4]))
            plt.imshow(image_np)
            plt.show()
            plt.close()
    
        # Keeping only worms which scored > 0.5
        found_worms = np.where(output_dict['detection_scores'] > 0.5)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][found_worms]
        output_dict['detection_classes'] = output_dict['detection_classes'][found_worms]
        output_dict['detection_scores'] = output_dict['detection_scores'][found_worms]
        output_dict['detection_masks'] = output_dict['detection_masks'][found_worms]
        output_dict['skeletons'] = []
        output_dict['frame_num'] = fName
        
        for m in output_dict['detection_masks']:
            output_dict['skeletons'].append(skeletonize(m).astype(np.uint8))
    
        
        OUTPUT_DIR_PATH = os.path.join('./data/inference_outputs', MODEL_NAME, dataset)
    
        
        if save_anns_to_file:
            #Save outputs to Pickle file
            os.makedirs(os.path.join(OUTPUT_DIR_PATH, 'annotations'), exist_ok=True)
            ANNS_OUTPUT_PATH = os.path.join(OUTPUT_DIR_PATH,'annotations', fName) + '.pickle'
            with open(ANNS_OUTPUT_PATH, 'wb') as fp:
                pickle.dump(output_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
                
        if save_overlays_to_file:
            #save image with annotations overlaid to file
            os.makedirs(os.path.join(OUTPUT_DIR_PATH, 'images'), exist_ok=True)
            IMG_OUTPUT_PATH = os.path.join(OUTPUT_DIR_PATH,'images', fName) + '.png'
            
            plt.figure(figsize=IMAGE_SIZE)
            
            # If the image ahsn't already been visualised, we need 
            # to add the masks and boxes now
            if not visualise_outputs:
                vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  output_dict['detection_boxes'],
                  output_dict['detection_classes'],
                  output_dict['detection_scores'],
                  category_index,
                  instance_masks=output_dict.get('detection_masks'),
                  use_normalized_coordinates=True,
                  line_thickness=2)
                
            plt.imshow(image_np)
            plt.axis('off')
            plt.savefig(fname=IMG_OUTPUT_PATH, bbox_inches='tight', pad_inches=0)
            plt.close
            
        
        output_dicts_list.append(output_dict)
        
    print("Finished analysing {} images.".format(NUM_IMAGES))
    mean_time = round(sum(inference_times[1:]) / (len(inference_times)-1), 3)
    fps = round((1/mean_time), 3)
    print("Average inference time for {} images: {}s ({} fps)".format(NUM_IMAGES, mean_time, fps))
    



