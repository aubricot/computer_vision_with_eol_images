# Utility functions for Bounding box coordinates display test
# Last modified on 8 September 2025

import pandas as pd
import numpy as np
import cv2
import urllib
from urllib.request import urlretrieve
from six.moves.urllib.request import urlopen

# Read in data file exported from "Combine output files A-D" block above
def read_datafile(fpath, sep="\t", header=0, disp_head=True):
    """
    Defaults to tab-separated data files with header in row 0
    """
    try:
        df = pd.read_csv(fpath, sep=sep, header=header)
        if disp_head:
          print("Data header: \n", df.head())
    except FileNotFoundError as e:
        raise Exception("File not found: Enter the path to your file in form field and re-run").with_traceback(e.__traceback__)
    
    return df

# Draw cropping box on image
def draw_box_on_image(df, img, i):
    # Get box coordinates
    xmin = df['xmin'][i].astype(int)
    ymin = df['ymin'][i].astype(int)
    xmax = df['xmin'][i].astype(int) + df['crop_width'][i].astype(int)
    ymax = df['ymin'][i].astype(int) + df['crop_height'][i].astype(int)
    boxcoords = [xmin, ymin, xmax, ymax]

    # Set box/font color and size
    maxdim = max(df['im_height'][i],df['im_width'][i])
    fontScale = maxdim/600
    box_col = (255, 0, 157)
  
    # Add label to image
    tag = df['class_name'][i]
    image_wbox = cv2.putText(img, tag, (xmin+7, ymax-12), cv2.FONT_HERSHEY_SIMPLEX, fontScale, box_col, 2, cv2.LINE_AA)  
  
    # Draw box label on image
    image_wbox = cv2.rectangle(img, (xmin, ymax), (xmax, ymin), box_col, 5)

    return image_wbox, boxcoords

# For uploading an image from url
# Modified from https://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv/
def url_to_image(url, timeout=10): # move to next image after 10 seconds
    try:
        resp = urllib.request.urlopen(url, timeout=timeout)
        image_data = resp.read()
        image_array = np.asarray(bytearray(image_data), dtype="uint8")
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image failed to load from URL: {url}\n")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image from URL: {url}\n{e}")
        return None