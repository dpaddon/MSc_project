#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified from the excellent pycococreator example at https://github.com/waspinator/pycococreator.git

"""

import datetime
import json
import os
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

#MODE = "ground_truth"
MODE = "tierpsy_evaluation"

ROOT_DIR = '/Users/daniel/Documents/UCL/Project/Data/annotation-data/cropped_collated_dataset'


OUTPUTS_DIR = '/Users/daniel/Documents/UCL/Project/Data/annotation-data/COCO_outputs'



def main():
    
    datasets = [f for f in os.listdir(ROOT_DIR) if not f.startswith('.') if not f.startswith('syn')]
    datasets = sorted(datasets)
    print("Datasets to be encoded as COCO JSON files: ")
    print(datasets)
    
    for DATASET in datasets:
        
        INFO = {
            "description": DATASET,
            "year": 2018,
            "contributor": "dpaddon",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }
        
        LICENSES = [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]
        
        CATEGORIES = [
            {
                'id': 1,
                'name': 'c_elegans',
                'supercategory': 'worm',
            },
]
        
        print("Processing {}".format(DATASET))
        
        IMAGE_DIR = os.path.join(ROOT_DIR, DATASET)
    
        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }
    
#        image_id = 1
        segmentation_id = 1
        
        
        #Get list of frame IDs
        filenames = sorted([f for f in os.listdir(IMAGE_DIR) if not f.startswith('.')], key=int)
    
        # go through each image
        for fname in filenames:
            image_id = fname
            image = Image.open(os.path.join(IMAGE_DIR, fname, 'image/image_{}.png'.format(fname)))
            image_info = pycococreatortools.create_image_info(
                image_id, fname, image.size)
            coco_output["images"].append(image_info)
    
    
                
            ANNOTATION_DIR = os.path.join(IMAGE_DIR, fname, 'masks')
            annotation_files = sorted([f for f in os.listdir(ANNOTATION_DIR) if not f.startswith('.')])
    
            # go through each associated annotations
            for annotation_filename in annotation_files:
                
    #            print(annotation_filename)
    
                category_info = {'id': 1, 'is_crowd': False}
                binary_mask = np.asarray(Image.open(os.path.join(ANNOTATION_DIR,annotation_filename))
                    .convert('1')).astype(np.uint8)
                
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=2)
                
    
                if annotation_info is not None:
                    #add score for COCO evaluation
                    annotation_info['score'] = 1.0
                    coco_output["annotations"].append(annotation_info)
    
                    segmentation_id = segmentation_id + 1
    
#            image_id = image_id + 1
    
        os.makedirs(os.path.join(OUTPUTS_DIR, MODE), exist_ok=True)
        with open('{}/{}/{}.json'.format(OUTPUTS_DIR, MODE, DATASET), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()