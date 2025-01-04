# Utility functions for running inference
# Last updated 3 January 2024 by K Wolcott
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

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
        max_predictions = 5
        stop = start + max_predictions
        cutoff = 5
    # To run for a larger set, use 5000 random images
    else:
        start = np.random.choice(a=N, size=1)[0]
        prediction_pool = 5500 # Many URLs broken, so initiate run for up to 5500 samples
        stop = start + prediction_pool
        cutoff = 5000

    return start, stop, cutoff

# To display loaded image with results
def imShow(fpath):
    image = cv2.imread(fpath)
    height, width = image.shape[:2]
    resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)
    fig = plt.gcf()
    fig.set_size_inches(9, 9)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.show()