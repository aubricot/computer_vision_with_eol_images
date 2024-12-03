# Utility functions for running inference
# Last updated 2 Dec 2024 by K Wolcott
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

# To read in EOL formatted data files
def read_datafile(fpath, sep="\t", header=0, disp_head=True, lineterminator='\n', encoding='latin1', dtype=None):
    try:
        df = pd.read_csv(fpath, sep=sep, header=header, lineterminator=lineterminator, encoding=encoding, dtype=dtype)
        if disp_head:
          print("Data header: \n", df.head())
    except FileNotFoundError as e:
        raise Exception("File not found: Enter the path to your file in form field and re-run").with_traceback(e.__traceback__)

    return df

# Define start and stop indices in EOL bundle for running inference
def set_start_stop(run, df):
    # To test with a tiny subset, use 5 random bundle images
    N = len(df)
    if "tiny subset" in run:
        start = np.random.choice(a=N, size=1)[0]
        max_predictions = 50
        stop = start + max_predictions
        cutoff = 5
    # To run for a larger set, use 500 random images
    else:
        start = np.random.choice(a=N, size=1)[0]
        prediction_pool = 10000 # Many URLs broken, so initiate run for up to 10000 samples, then stop loop once it hits 500 samples
        stop = start + prediction_pool
        cutoff = 500

    return start, stop, cutoff

# Load in image from URL
# Modified from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/saved_model.ipynb#scrollTo=JhVecdzJTsKE
def image_from_url(url, fn, pixels):
    file = tf.keras.utils.get_file(fn, url) # Filename doesn't matter
    disp_img = tf.keras.preprocessing.image.load_img(file)
    image = tf.keras.preprocessing.image.load_img(file, target_size=[pixels, pixels])
    colormode = image.getbands()
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image[tf.newaxis,...])

    return image, disp_img, colormode

# Get info from predictions to display on images
def get_predict_info(predictions, url, i, stop, start, dataset_labels):
    # Get info from predictions
    label_num = np.argmax(predictions[0], axis=-1)
    conf = predictions[0][label_num]
    im_class = dataset_labels[label_num]
    
    return label_num, conf, im_class

# To display loaded image with results
def plot_image_results(i, disp_img, tag):
        _, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(disp_img)
        plt.axis('off')
        plt.title("{}) Image type: {} ".format(i+1, tag))