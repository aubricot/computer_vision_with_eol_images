# Utility functions for dataset preprocessing - Classification for image tagging
# Last updated 7 Aug 2025 by K Wolcott
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

# To read in EOL formatted data files
def read_datafile(fpath, sep="\t", header=0, disp_head=True):
    try:
        df = pd.read_csv(fpath, sep=sep, header=header)
        if disp_head:
          print("Data header: \n", df.head())
    except FileNotFoundError as e:
        raise Exception("File not found: Enter the path to your file in form field and re-run").with_traceback(e.__traceback__)

    return df

# To display an image already loaded into the runtime
def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)

# Define start and stop indices in EOL bundle for running inference
def set_start_stop(run, df):
    # To test with a tiny subset, use 50 random bundle images
    N = len(df)
    if "tiny subset" in run:
        start=np.random.choice(a=N, size=1)[0]
        stop=start+50
    # Run for all images
    else:
        start=0
        stop=N

    return start, stop

# Define image augmentation pipeline
# modified from https://github.com/aleju/imgaug
def augment_image(image):
    seq = iaa.Sequential([
            iaa.Crop(px=(1, 16), keep_size=False), # crop by 1-16px, resize resulting image to orig dims
            iaa.Affine(rotate=(-25, 25)), # rotate -25 to 25 degrees
            iaa.GaussianBlur(sigma=(0, 3.0)), # blur using gaussian kernel with sigma of 0-3
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True)
            ])
    # Optional: set seed to make augmentations reproducible across runs, otherwise will be random each time
    ia.seed(1)
    # Augment image
    image_aug = seq.augment(image=image)

    return image_aug

# Filter by rating of interest
def filter_by_rating(df, filter=filter, disp_head=False):
    rating = df.loc[round(df["overall_rating"])==int(filter)]
    rating = rating["obj_url"].copy()
    
    if disp_head:
          print("Rating = {}}:\n {}".format(filter, rating.head()))
    print("\n Number of available ratings for training/testing class {}: \n {}".format(filter, len(rating)))

    return rating
