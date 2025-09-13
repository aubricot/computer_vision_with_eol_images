# Utility functions for running inference - Classification for image tagging
# Last updated 13 Sep 2025 by K Wolcott
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import csv

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
        cutoff = 5
        N = N - cutoff
        start = np.random.choice(a=N, size=1)[0]
        max_predictions = 50
        stop = start + max_predictions

    # To run for a larger set, use 500 random images
    else:
        cutoff = 1000
        N = max(N - cutoff, 0)
        start = np.random.choice(a=N, size=1)[0]
        prediction_pool = 10000 # Many URLs broken, so initiate run for up to 10000 samples, then stop loop once it hits 500 samples
        stop = start + prediction_pool


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

# Run inference
def run_inference(model, df, pixels, dataset_labels, start, stop, cutoff, outfpath, image_from_url, get_predict_info, export_results):
    all_predictions = []

    for i, row in enumerate(df.iloc[start:stop].iterrows()):
        try:
            url = df['eolMediaURL'][i]
            img, _, _ = image_from_url(url, f"{i}.jpg", pixels)

            start_time = time.time()
            predictions = model.predict(img, batch_size=1)
            label_num, conf, det_imclass = get_predict_info(predictions, url, i, stop, start, dataset_labels)
            end_time = time.time()

            export_results(df, url, det_imclass, conf, i)

            print(f"\033[92mCompleted for {len(all_predictions)} of {cutoff} files\033[0m")
            all_predictions.append(det_imclass)

            if len(all_predictions) >= cutoff:
                break

        except Exception as e:
            print(f"\033[91mError for URL: {url} -- {str(e)}\033[0m")

    print("\n\n~~~\n\033[92mInference complete!\033[0m \033[93mRun these steps for remaining batches before proceeding.\033[0m\n~~~")


# Run inference
def run_inference_imtype(model, df, pixels, dataset_labels, start, stop, cutoff, outfpath, image_from_url, get_predict_info, export_results, cartoonize=False, calc_img_diffs=False):
    all_predictions = []

    for i, row in enumerate(df.iloc[start:stop].iterrows()):
        try:
            url = df['eolMediaURL'][i]
            img, disp_img, _ = image_from_url(url, f"{i}.jpg", pixels)
            
            if cartoonize:
                 # Cartoonization
                img_cv = np.array(disp_img) # For working with cv2 lib
                img2 = cartoonize(img_cv)
                # Calculate differences between original and cartoonized image
                mnorm_pp, _, _, _ = calc_img_diffs(img_cv, img2)

            start_time = time.time()
            predictions = model.predict(img, batch_size=1)
            label_num, conf, det_imclass = get_predict_info(predictions, url, i, stop, start, dataset_labels)
            end_time = time.time()

            export_results(df, url, det_imclass, mnorm_pp, conf, i)

            print(f"\033[92mCompleted for {len(all_predictions)} of {cutoff} files\033[0m")
            all_predictions.append(det_imclass)

            if len(all_predictions) >= cutoff:
                break

        except Exception as e:
            print(f"\033[91mError for URL: {url} -- {str(e)}\033[0m")

    print("\n\n~~~\n\033[92mInference complete!\033[0m \033[93mRun these steps for remaining batches before proceeding.\033[0m\n~~~")