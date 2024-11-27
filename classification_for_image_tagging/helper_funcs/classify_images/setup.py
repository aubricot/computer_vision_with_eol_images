# Set up directory structure and model parameters
# Last updated 29 June 2023 by K Wolcott
import os
import subprocess
import shutil
from pathlib import Path
import tensorflow as tf

# Set up directory structure & make darknet
def setup_dirs(cwd):
        # Make folders if they don't already exist
        if not os.path.exists(cwd):
                os.makedirs(cwd)

# Get info about trained classification model
def unpack_EOL_model(use_EOL_model, saved_models_dir, basewd, TRAIN_SESS_NUM, classif_type):
        # Use EOL pre-trained model
        if use_EOL_model:
                # Unpack saved model files and set up directory structure
                if not os.path.exists(saved_models_dir):
                        # Make folder for trained model
                        trained_model_dir = saved_models_dir + TRAIN_SESS_NUM + '/'
                        os.makedirs(trained_model_dir)
                        # Unzip saved model files
                        zipped_model_fn = '/content/' + TRAIN_SESS_NUM + '.zip'
                        shutil.unpack_archive(zipped_model_fn, TRAIN_SESS_NUM)
                        zipped_model_dir = basewd + '/' + TRAIN_SESS_NUM + '/content/drive/MyDrive/summer20/classification/' + classif_type + '/saved_models/' + TRAIN_SESS_NUM + '/'
                        fns = os.listdir(zipped_model_dir)
                        for fn in fns:
                                shutil.move(os.path.join(zipped_model_dir, fn), trained_model_dir)
                        shutil.rmtree(basewd + '/' + TRAIN_SESS_NUM + '/')
                        os.remove(zipped_model_fn)
    
        return trained_model_dir

# Load saved model from directory
def load_saved_model(trained_model_dir, module_selection):
        # Load trained model from path
        model = tf.keras.models.load_model(trained_model_dir)
        # Get name and image size for model type
        handle_base, pixels = module_selection

        return model, pixels, handle_base