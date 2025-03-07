# Functions to set up directory structure and load models - Aves
# Last updated 6 Feb 2025 by K Wolcott
import os
import subprocess
import tensorflow as tf
import tensorflow_hub as hub
import json

# Set up directory structure & build Tensorflow Obj Det API
def setup_dirs(cwd):
        # Make folders if they don't already exist
        if not os.path.exists(cwd):
                os.makedirs(cwd)
                os.chdir(cwd)
                os.mkdir('tf_models')
                os.makedirs('results/inspect_results')
                os.chdir('tf_models')

        return cwd

# Load Tensorflow Hub pre-trained object detector
def load_tfhub_detector(model):
    if 'SSD' in model:
        module_handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
        mod_abbv = '_ssd'
    elif 'RCNN' in model:
        module_handle = "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1"
        mod_abbv = '_rcnn'
    else:
        print('Error: Model name does not correspond to module_handle')
    print('Loading {} from TF-Hub...'.format(model))
    detector = hub.load(module_handle)

    return detector, module_handle, mod_abbv

# Convert labelmap.json to dictionary for inference
def convert_labelmap(fpath = 'labelmap.json'):
    with open(fpath, "r") as f:
        # Load the dictionary from the file
        label_map = json.load(f)
    label_map = {int(k):str(v) for k,v in label_map.items()}
    
    return label_map