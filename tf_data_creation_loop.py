import tensorflow as tf
from matplotlib.image import imread
import os
import numpy as np
import io
import PIL.Image
import matplotlib.pyplot as plt
from random import shuffle
import csv




from tensorflow.models.research.object_detection.utils import dataset_util

CWD = os.getcwd()

abs_path = '/Users/daniel/Documents/UCL/Project/Data/annotation-data/cropped_collated_dataset'
#abs_path = '/Users/daniel/Documents/UCL/Project/Data/'
OUTPUT_PATH = '/Users/daniel/Documents/UCL/Project/Data/cropped_tf_record'

flags = tf.app.flags
flags.DEFINE_string('output_path', OUTPUT_PATH, 'Path to output TFRecords')
FLAGS = flags.FLAGS

val_size = 0.2


def create_tf_example(path, frame_num):
    
    """Creates a tf.Example proto for an individual image and its masks.

      Args:
          path: path to directory containing all images to be encoded
          frame_num: The frame number of the image to be encoded.
    
      Returns:
          example: The created tf.Example.
      """
    # TODO(user): Populate the following variables from your example.
     
    
    img_path = path + '/' + str(frame_num) + '/image/image_' + str(frame_num) + '.png'
    img = imread(img_path)
#    plt.imshow(img)
      
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_png = fid.read()
        encoded_png_io = io.BytesIO(encoded_png)
#        image = PIL.Image.open(encoded_png_io)

          
    height = int(img.shape[0]) # Image height
    width = int(img.shape[1]) # Image width
#    print("Height: {}, width: {}".format(height, width))
    filename = str(frame_num) # + '.png' # Filename of the image. Empty if image is not from file
#    encoded_image_data = img.tobytes() # Encoded image bytes
    image_format = b'png' # b'jpeg' or b'png'
    
    
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
               # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
               # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    
#    masks = []
    encoded_mask_png_list = []
    
    for mask_id in (os.listdir(path + '/' + frame_num + '/masks/')):
#        mask_img = imread(path + '/' + frame_num + '/masks/' + mask_id)
#        mask_img = cv2.convertScaleAbs(mask_img)
#        mask_img, contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#        x, y, w, h = cv2.boundingRect(contours[0])
        
#        mask_img = imread(path + '/' + frame_num + '/masks/' + mask_id)[:, :, :3]
#        indices = np.where(mask_img)
        
        mask_path = path + '/' + frame_num + '/masks/' + mask_id
        
        with tf.gfile.GFile(mask_path, 'rb') as fid:
            encoded_mask_png = fid.read()
            encoded_png_io = io.BytesIO(encoded_mask_png)
            mask = PIL.Image.open(encoded_png_io)
            
        
        mask_np = np.asarray(mask)[:,:,:3]
#        print(mask_np.shape)
        indices = np.where(mask_np)
        
        
#        print("Indices: ", indices)
        
        if indices[0].size > 0:        
            xmin = float(np.min(indices[1]))
            xmax = float(np.max(indices[1]))
            ymin = float(np.min(indices[0]))
            ymax = float(np.max(indices[0]))   

            
            xmins.append(xmin/width)
            xmaxs.append(xmax/width)
            
            ymins.append(ymin/height)
            ymaxs.append(ymax/height)
            
            classes_text.append('c_elegans'.encode('utf8'))
            classes.append(int(1))
        
            mask_remapped = mask_np.astype(np.uint8)
#            masks.append(mask_remapped)
            mask_img = PIL.Image.fromarray(mask_remapped)
            
#            plt.imshow(mask_img)
#            plt.show()
            
            output = io.BytesIO()
            mask_img.save(output, format='PNG')
            encoded_mask_png_list.append(output.getvalue())
            
        else:
            pass

#    encoded_mask_png_list = []
#    for mask in masks:
#        mask_img = PIL.Image.fromarray(mask)
#        output = io.BytesIO()
#        mask_img.save(output, format='PNG')
#        encoded_mask_png_list.append(output.getvalue())
        
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/mask': dataset_util.bytes_list_feature(encoded_mask_png_list),
    }))
    return tf_example


def main(_):
    
    datasets = [f for f in os.listdir(abs_path) if not f.startswith('.')]
    datasets = sorted(datasets)
    print("Datasets to be encoded as tf.records: ")
    print(datasets)
    
    for d_s in datasets:
        
        dataset_path = os.path.join(abs_path, d_s)
        print(dataset_path)
        frames = [f for f in os.listdir(dataset_path) if not f.startswith('.')][:250]
#        print(frames)
        
        # Commented out shuffling for tracking - tracking requires sequential frames, 
        # so the frames are kept in order so that the validation set is one contiguous block
        # the frames within each record are shuffled by TF during training anyway
        
#        shuffle(frames)
        num_frames = len(frames)
        
        
        #train set
        train_num = int(num_frames * (1-val_size))
        print("Forming train tf.record shard of {} images".format(train_num))
        
        train_frames = frames[:train_num]
        writer = tf.python_io.TFRecordWriter(FLAGS.output_path + '/train/train_{}.record'.format(d_s))

        for frame_num in train_frames:
#            print(frame_num)
            tf_example = create_tf_example(abs_path + '/' + d_s, frame_num)
            writer.write(tf_example.SerializeToString())
          
        writer.close()
        

        # eval set
        print("Forming eval tf.record shard of {} images".format(num_frames-train_num))
        val_frames = frames[train_num:]
        writer = tf.python_io.TFRecordWriter(FLAGS.output_path + '/eval/eval_{}.record'.format(d_s))

        for frame_num in val_frames:
#            print(frame_num)
            tf_example = create_tf_example(abs_path + '/' + d_s, frame_num)
            writer.write(tf_example.SerializeToString())
          
        writer.close()
        

        # Save a CSV with the filenames of the training and validation frames
        split_csv = os.path.join(OUTPUT_PATH, '{}_splits.csv'.format(d_s))
        with open(split_csv, 'w') as trainfile:
            wr = csv.writer(trainfile, quoting=csv.QUOTE_ALL)
            wr.writerow(sorted(train_frames))
            wr.writerow(sorted(val_frames))


if __name__ == '__main__':
  tf.app.run()