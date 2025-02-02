# Functions to set up directory structure and load models - Lepidoptera
# Last updated 1 February 2025 by K Wolcott
import os
import subprocess
import shutil
import tensorflow as tf
import sys
sys.path.append("tf_models/models/research/")


# Set up directory structure & build Tensorflow Obj Det API
def setup_dirs(cwd):
        # Make folders if they don't already exist
        if not os.path.exists(cwd):
                os.makedirs(cwd)
                os.chdir(cwd)
                os.mkdir('tf_models')
                os.makedirs('results/inspect_results')
                os.chdir('tf_models')
                # Clone the Tensorflow Model Garden
                repoPath = 'https://github.com/tensorflow/models'
                subprocess.check_call(['git', 'clone', repoPath, '--depth=1'])

# Get info about trained classification model
def unpack_EOL_model(use_EOL_model, saved_models_dir, PATH_TO_CKPT, cwd):
        # Use EOL pre-trained model
        if use_EOL_model:
                # Unpack saved model files and set up directory structure
                if not os.path.exists(saved_models_dir):
                        os.chdir(cwd)
                        # Download labelmap.pbtxt
                        subprocess.check_call(['gdown', '1V7PzevCvMMw6BXXknVbMuKANsm023nYw']) 
                        os.makedirs('tf_models/train_demo/rcnn/finetuned_model')
                        os.chdir('tf_models/train_demo/rcnn/finetuned_model')
                        # Download frozen_inference_graph.pb
                        subprocess.check_call(['gdown', '1DfdZXFwEuCHt8htJ-o4brY9-p65kfELW']) 
                        os.chdir(cwd)

                else:
                        print("\033[93m trained_models_dir already exists at: \033[0m", saved_models_dir)
        
                # Restore frozen detection graph (trained model)    
                print("\nLoading trained model from: \n", PATH_TO_CKPT)
                detection_graph = tf.Graph()
                with detection_graph.as_default():
                        od_graph_def = tf.compat.v1.GraphDef()
                        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                                serialized_graph = fid.read()
                                od_graph_def.ParseFromString(serialized_graph)
                                tf.import_graph_def(od_graph_def, name='')

        else:
                detection_graph = None
                print("\033[93m use_EOL_model set to False. Adjust parameters if using your own custom model or an EOL model.\033[0m")

        return detection_graph