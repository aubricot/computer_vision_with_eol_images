# Set up directory structure and model parameters
# Last updated 2 Dec 2024 by K Wolcott
import os
import subprocess
import shutil
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras

# Set up directory structure & make darknet
def setup_dirs(cwd):
        # Make folders if they don't already exist
        if not os.path.exists(cwd):
                os.makedirs(cwd)

# Get info about model based on training attempt number
def get_model_info(TRAIN_SESS_NUM):
    # Session 18 - ratings
    if int(TRAIN_SESS_NUM) == 18:
        module_selection =("mobilenet_v2_1.0_224", 224)
        dataset_labels = ['bad', 'good'] # Classes aggregated after attempt 7: 1/2 -> bad, 4/5 -> good
    # Session 20 - ratings
    elif int(TRAIN_SESS_NUM) == 20:
        module_selection = ("inception_v3", 299)
        dataset_labels = ['bad', 'good'] # Classes aggregated after attempt 7: 1/2 -> bad, 4/5 -> good
    # Session 6 - ratings
    elif int(TRAIN_SESS_NUM) == 6:
        module_selection = ("inception_v3", 299)
        dataset_labels = ['1', '2', '3', '4', '5'] # Before aggregating classes

    # Session 11 - image type
    elif int(TRAIN_SESS_NUM) == 11:
        module_selection = ("inception_v3", 299)
        dataset_labels = ['herb', 'illus', 'map', 'null', 'phylo']
    
    # Session 13 - image type
    elif int(TRAIN_SESS_NUM) == 13:
        module_selection = ("mobilenet_v2_1.0_224", 224)
        dataset_labels = ['herb', 'illus', 'map', 'null', 'phylo']

    return module_selection, dataset_labels

# Get info about trained classification model
def unpack_EOL_model(use_EOL_model, trained_model_dir, basewd, TRAIN_SESS_NUM, classif_type):
    # Use EOL pre-trained model
    if use_EOL_model:
        # Unpack saved model files and set up directory structure
          if not os.path.exists(trained_model_dir):
              # Make folder for trained model
              os.makedirs(trained_model_dir)
              # Unzip saved model files
              zipped_model_fn = '/content/' + TRAIN_SESS_NUM + '.zip'
              shutil.unpack_archive(zipped_model_fn, TRAIN_SESS_NUM)
              zipped_model_dir = TRAIN_SESS_NUM + '/content/drive/MyDrive/summer20/classification/' + classif_type + '/saved_models/' + TRAIN_SESS_NUM + '/'
              fns = os.listdir(zipped_model_dir)
              for fn in fns:
                  shutil.move(os.path.join(zipped_model_dir, fn), trained_model_dir)
              shutil.rmtree(TRAIN_SESS_NUM + '/')
              os.remove(zipped_model_fn)
              print("\033[92m Model {} successfully unpacked at {}".format(TRAIN_SESS_NUM, trained_model_dir))

          else:
              print("\033[93m trained_models_dir already exists at: ", trained_model_dir)
    else:
        print("\033[93m use_EOL_model set to False. Adjust parameters if using your own custom model or an EOL model.")

# Load saved model from directory
def load_saved_model(models_wd, TRAIN_SESS_NUM, module_selection):
    # Load saved/pre-trained model from path
    saved_model_path = models_wd + '/' + TRAIN_SESS_NUM
    # Load the SavedModel as a tf hub layer (Keras 3 patch)
    model = tf_keras.Sequential([hub.KerasLayer(saved_model_path)])
    # Get name and image size for model type
    handle_base, pixels = module_selection
    print("\n\033[92m Successfully loaded model {}\033[0m from: \n{} \n for training attempt {} with input image size of {},{} pixels".format(handle_base, saved_model_path, TRAIN_SESS_NUM, pixels, pixels))

    return model, pixels, handle_base