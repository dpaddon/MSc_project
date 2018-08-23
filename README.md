# MSc Project - accurate segmentation of c. elegans using Mask-RCNN.

## Instructions for use

### Creating a set of .png images and their associated masks (one png mask per worm) from an HDF5 file and XML hand-drawn annotations.

For this task, use the file annotation_data_generation.py


### Synthetic data generation

Supplementary training data, with individual worms added in to images to mimic the conditions of the failure cases of the Tierpsy tracker, can be generated within the synthetic_data_generator.ipynb python notebook.

### Cropping data for training

The original images as recorded by the camera are 2048*2048 pixels. Training on these images exceeds the memory capacity of a typical Titan X GPU (11GB). As such, for this investigation the images were cropped into 16 smaller tiles of 512*512px each. 

This is automatically performed by the annotation_data_generation.py file, but for training on existing data this can be performed with the data_cropper.py file.


### Creating TensorFlow records for training

The TensorFlow models used for training take data in the tf record file format. The tf_data_creation_loop.py file will run through a set of datasets, creating a tf record "shard" of training and validation data from each set.

Models can be trained on a list of shards, specified from within the config file. 



