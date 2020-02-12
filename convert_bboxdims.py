# Converting object detection bounding box coordinates to EOL crop format
# Last modified 10 February 20

import csv
import numpy as np
import pandas as pd
import os

# Read in sample crops file exported from object_detection_for_image_cropping_yolo.ipynb
df = pd.read_csv('chiroptera_det_crops_20000.tsv', sep='\t', header=0)
print(df.head())

# Correct for images with 1+ bounding boxes by making a 'super box' containing all small boxes per image
# For each image, take smallest xmin, ymin and largest xmax, ymax from bbox coordinates
xmin_unq = pd.DataFrame(df.groupby(['image_url'])['xmin'].min())
ymin_unq = pd.DataFrame(df.groupby(['image_url'])['ymin'].min())
xmax_unq = pd.DataFrame(df.groupby(['image_url'])['xmax'].max())
ymax_unq = pd.DataFrame(df.groupby(['image_url'])['ymax'].max())
print(xmin_unq.head())

# Workaround to get im_height and im_width in same format/order as 'super box' coords
# There is only one value for im_height and im_width, so taking max will return unchanged values
im_height_unq = pd.DataFrame(df.groupby(['image_url'])['im_height'].max())
im_width_unq = pd.DataFrame(df.groupby(['image_url'])['im_width'].max())

# Make a new dataframe with the unique values (ie. for only 1 bbox/image)
crops_unq = im_height_unq.merge(im_width_unq, left_index=True, right_index=True)
crops_unq = crops_unq.merge(xmin_unq, left_index=True, right_index=True)
crops_unq = crops_unq.merge(ymin_unq, left_index=True, right_index=True)
crops_unq = crops_unq.merge(xmax_unq, left_index=True, right_index=True)
crops_unq = crops_unq.merge(ymax_unq, left_index=True, right_index=True)
print(crops_unq.head())

# Calculate new crop height and width using unique bounding box values
crops_unq['crop_height'] = crops_unq['ymax'] - crops_unq['ymin']
crops_unq['crop_width'] = crops_unq['xmax'] - crops_unq['xmin']
print(crops_unq.head())

# Get EOL identifiers from eolMediaURLs
## Change 1st col from index to normal data column
crops_unq.reset_index(inplace=True)
crops_unq.rename(columns={'image_url': 'eolMediaURL'}, inplace=True)

## Get dataObjectVersionIDs and identifiers from 1st 2 and 2nd to last cols of EOL breakdown file 
df = pd.read_csv('images_for_Chiroptera_20K_breakdown_000001.txt', sep='\t', header=0)
df = df.iloc[:, np.r_[0:2,-2]]
print(df.head())

## Map dataObjectVersionIDs to crops_unq using identifiers as the index
crops_unq.set_index('eolMediaURL', inplace=True, drop=True)
df.set_index('eolMediaURL', inplace=True, drop=True)
crops = crops_unq.merge(df, left_index=True, right_index=True)
print(crops.head())

# Convert bounding box/cropping dimensions to square and make sure crop boxes aren't out of image bounds
for i, row in crops.iterrows():
    # When crop height > crop width:
    if crops.crop_height[i] > crops.crop_width[i]:
        # If crop dimensions are smaller than image dimensions (ie. not out of bounds), make crop width = crop height
        if crops.crop_height[i] <= min(crops.im_height[i], crops.im_width[i]):
            # Center position of transformed crop
            crops.xmin[i] = crops.xmin[i] - 0.5*(crops.crop_height[i] - crops.crop_width[i])
            # Make crop dimensions square
            crops.crop_width[i] = crops.crop_height[i]
            crops.crop_height[i] = crops.crop_height[i]
    
        # If padded crop dimensions are larger than image dimensions (ie. out of bounds), make crop width & height = image height or width (smaller dimension)
        else:
            # Center position of transformed crop
            crops.xmin[i] = crops.xmin[i] - 0.5*(min(crops.im_height[i], crops.im_width[i]) - crops.crop_width[i])
            # Set crop dimensions equal to smaller image dimension (height or width)
            crops.crop_width[i] = min(crops.im_height[i], crops.im_width[i])
            crops.crop_height[i] = min(crops.im_height[i], crops.im_width[i])
    
    # When crop width > crop height
    else:
        # If crop dimensions are smaller than image dimensions (ie. not out of bounds), set crop height = padded crop width        
        if crops.crop_width[i] <= min(crops.im_width[i], crops.im_height[i]):
            # Center position of transformed crop
            crops.ymin[i] = crops.ymin[i] - 0.5*(crops.crop_width[i] - crops.crop_height[i])
            # Make crop dimensions square
            crops.crop_height[i] = crops.crop_width[i]
            crops.crop_width[i] = crops.crop_width[i]
    
        # If crop dimensions are larger than image dimensions (ie. out of bounds), make crop width & height = image height or width (smaller dimension)
        else:
            # Center position of transformed crop
            crops.ymin[i] = crops.ymin[i] - 0.5*(min(crops.im_height[i], crops.im_width[i]) - crops.crop_height[i])
            # Set crop dimensions equal to image dimensions
            crops.crop_width[i] = min(crops.im_height[i], crops.im_width[i])
            crops.crop_height[i] = min(crops.im_height[i], crops.im_width[i])

## Check that crop position isn't outside of image bounds, if so, make xmin or ymin=0
for i, row in crops.iterrows():
        if crops.ymin[i] + crops.crop_height[i] >= crops.im_height[i]:
                crops.ymin[i] = 0
        elif crops.xmin[i] + crops.crop_width[i] >= crops.im_width[i]:
                crops.xmin[i] = 0

## Image coordinates should be positive, set negative xmin and ymin values to 0
crops.xmin[crops.xmin < 0] = 0
crops.ymin[crops.ymin < 0] = 0

print(crops.head())

# Make new xmax and ymax values by adding crop dims to xmin and ymin
for i, row in crops.iterrows():
    crops.xmax[i] = crops.xmin[i] + crops.crop_width[i]
    crops.ymax[i] = crops.ymin[i] + crops.crop_height[i]

# Test that dimensions were modified appropriately for dataset by exporting crop coordinates to display_test.tsv 
# Load this file into crop_coords_display_test.ipynb and visualize results
crops.to_csv('chiroptera_crops_rcnn_20000img_display_test.tsv', sep='\t', index=True)

# Get image and cropping dimensions in EOL format (concatenated string with labels)
# {"height":"423","width":"640","crop_x":123.712,"crop_y":53.4249,"crop_width":352,"crop_height":0}
crops['crop_dimensions'] = np.nan
for i, row in crops.iterrows():
    crops.crop_dimensions[i] = ('{{"height":"{}","width":"{}","crop_x":{},"crop_y":{},"crop_width":{},"crop_height":{}}}'
    .format(crops.im_height[i], crops.im_width[i], crops.xmin[i], crops.ymin[i], crops.crop_width[i], crops.crop_height[i]))
crops.reset_index(inplace=True)
print(crops.head())

# Create EOL crops formatted dataframe from cols: identifier, dataobjectversionid, eolmediaurl, and crop_dimensions
eol_crops = pd.DataFrame(crops.iloc[:,np.r_[-3,-2,0,-1]])
print(eol_crops.head())


# Write results to tsv formmatted to EOL crop coordinate standards
eol_crops.to_csv('chiroptera_crops_rcnn_20000img.tsv', sep='\t', index=False)