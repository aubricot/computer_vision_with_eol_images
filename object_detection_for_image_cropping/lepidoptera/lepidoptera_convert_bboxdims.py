# Converting object detection bounding box coordinates to EOL crop format
# Last modified 22 March 20

import time
start = time.time()
import csv
import numpy as np
import pandas as pd
import os

# Read in sample crops file exported from object_detection_for_image_cropping_yolo.ipynb
crops = pd.read_csv('object_detection_for_image_cropping/data_files/input/Lepidoptera/lepidoptera_det_crops_20000.tsv', sep='\t', header=0)
print(crops.head())

# Correct for images with 1+ bounding boxes by making a 'super box' containing all small boxes per image
# For each image, take smallest xmin, ymin and largest xmax, ymax from bbox coordinates
xmin_unq = pd.DataFrame(crops.groupby(['image_url'])['xmin'].min())
ymin_unq = pd.DataFrame(crops.groupby(['image_url'])['ymin'].min())
xmax_unq = pd.DataFrame(crops.groupby(['image_url'])['xmax'].max())
ymax_unq = pd.DataFrame(crops.groupby(['image_url'])['ymax'].max())
print(xmin_unq.head())

# Workaround to get im_height and im_width in same format/order as 'super box' coords
# There is only one value for im_height and im_width, so taking max will return unchanged values
im_height_unq = pd.DataFrame(crops.groupby(['image_url'])['im_height'].max())
im_width_unq = pd.DataFrame(crops.groupby(['image_url'])['im_width'].max())

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
bd = pd.read_csv('object_detection_for_image_cropping/data_files/input/Lepidoptera/images_for_Lepidoptera_20K_breakdown_000001.txt', sep='\t', header=0)
bd = bd.iloc[:, np.r_[0:2,-2]]
print(bd.head())

## Map dataObjectVersionIDs to crops_unq using identifiers as the index
crops_unq.set_index('eolMediaURL', inplace=True, drop=True)
bd.set_index('eolMediaURL', inplace=True, drop=True)
df = crops_unq.merge(bd, left_index=True, right_index=True)
print(df.head())

# Convert bounding box/cropping dimensions to square, add padding, and make sure crop boxes aren't out of image bounds
for i, row in df.iterrows():
    # Optional: Pad by xx% larger crop dimension (height)
    pad = 0 * max(df.crop_height[i], df.crop_width[i])
    df.ymin[i] = df.ymin[i] - pad
    df.xmin[i] = df.xmin[i] - pad
    # Assign crop height and width values for use filtering data through loops below
    crop_h = df.crop_height[i]
    crop_w = df.crop_width[i]

    # Crop Height > Crop Width
    # See project wiki "Detailed explanation with drawings: convert_bboxdims.py", Scenario 1
    # Where crop height is greater than crop width and is smaller than image dimensions
    if crop_h > crop_w and \
    (crop_h + pad) <= min(df.im_height[i], df.im_width[i]):
        # Make new crop dimensions square by setting crop width equal to crop height
        df.crop_height[i] = df.crop_height[i] + pad
        df.crop_width[i] = df.crop_height[i]       
        # X and Y dims not OOB (out of bounds)
        if (df.xmax[i] + 0.5*(crop_h - crop_w) + pad) <= df.im_width[i] and \
        (df.ymax[i] + pad) <= df.im_height[i]:
            df.xmin[i] = df.xmin[i] - 0.5*(crop_h - crop_w)
        # X dims OOB (+), shift cropping box left
        elif (df.xmax[i] + 0.5*(crop_h - crop_w) + pad) > df.im_width[i] and \
        (df.ymax[i] + pad) <= df.im_height[i]:
            df.xmin[i] = df.xmin[i] - (df.xmax[i] + 0.5*(crop_h - crop_w) + pad - df.im_width[i])
        # Y dims OOB (+), shift cropping box down
        elif (df.xmax[i] + 0.5*(crop_h - crop_w) + pad) <= df.im_width[i] and \
        (df.ymax[i] + pad) > df.im_height[i]:  
            df.xmin[i] = df.xmin[i] - 0.5*(crop_h - crop_w)
            df.ymin[i] = df.ymin[i] - (df.ymax[i] + pad - df.im_height[i])
        # X and Y dims OOB (+), shift cropping box down and left   
        elif (df.xmax[i] + 0.5*(crop_h - crop_w) + pad) > df.im_width[i] and \
        (df.ymax[i] + pad) > df.im_height[i]:
            df.xmin[i] = df.xmin[i] - (df.xmax[i] + 0.5*(crop_h - crop_w) + pad - df.im_width[i])
            df.ymin[i] = df.ymin[i] - (df.ymax[i] + pad - df.im_height[i])
    # Where crop height is greater than crop width, but is not within image dimensions	
    elif crop_h > crop_w and \
    (crop_h + pad) > min(df.im_height[i], df.im_width[i]):
        # Make square by setting crop dims equal to smaller image dim (width)
        df.crop_width[i] = df.im_width[i]
        df.crop_height[i] = df.im_width[i] 
        # Make crop x dims equal to image x dims
        df.xmin[i] = 0
        # Center crop x dims
        df.ymin[i] = df.ymin[i] + 0.5*(crop_h - df.im_width[i]) 
    # Where crop height is greater than crop width, but neither is smaller than image dimensions
    elif crop_h > crop_w and \
    min(crop_h, crop_w) <= min(df.im_height[i], df.im_width[i]):
        # Do not crop, set values equal to image dimensions
        df.crop_width[i] = df.im_width[i]
        df.crop_height[i] = df.im_height[i] 
        df.ymin[i] = 0
        df.xmin[i] = 0   
    
    # Crop Width > Crop Height
    # See project wiki "Detailed explanation with drawings: convert_bboxdims.py", Scenario 2
    # Where crop width is greater than crop height and is smaller than image dimensions
    elif crop_w > crop_h and \
    (crop_w + pad) <= min(df.im_height[i], df.im_width[i]):
        # Make new crop dimensions square by setting crop height equal to crop width
        df.crop_width[i] = df.crop_width[i] + pad  
        df.crop_height[i] = df.crop_width[i]   
        # X and Y dims not OOB (out of bounds)
        if (df.ymax[i] + 0.5*(crop_w - crop_h) + pad) <= df.im_height[i] and \
        (df.xmax[i] + pad) <= df.im_width[i]:
            df.ymin[i] = df.ymin[i] - 0.5*(crop_w - crop_h)
        # X dims OOB (+), shift cropping box left
        elif (df.ymax[i] + 0.5*(crop_w - crop_h) + pad) > df.im_height[i] and \
        (df.xmax[i] + pad) <= df.im_width[i]:
            df.ymin[i] = df.ymin[i] - (df.ymax[i] + 0.5*(crop_w - crop_h) + pad - df.im_height[i])
        # Y dims OOB (+), shift cropping box down
        elif (df.ymax[i] + 0.5*(crop_w - crop_h) + pad) <= df.im_height[i] and \
        (df.xmax[i] + pad) > df.im_width[i]:  
            df.ymin[i] = df.ymin[i] - 0.5*(crop_w - crop_h)
            df.xmin[i] = df.xmin[i] - (df.xmax[i] + pad - df.im_width[i])
        # X and Y dims OOB (+), shift cropping box down and left   
        elif (df.ymax[i] + 0.5*(crop_w - crop_h) + pad) > df.im_height[i] and \
        (df.xmax[i] + pad) > df.im_width[i]:
            df.ymin[i] = df.ymin[i] - (df.ymax[i] + 0.5*(crop_w - crop_h) + pad - df.im_height[i])
            df.xmin[i] = df.xmin[i] - (df.xmax[i] + pad - df.im_width[i])
    # Where crop width is greater than crop height, but is not within image dimensions
    elif crop_w > crop_h and \
    (crop_w + pad) > min(df.im_height[i], df.im_width[i]):
        # Make square by setting crop dims equal to smaller image dim (height)
        df.crop_width[i] = df.im_height[i]
        df.crop_height[i] = df.im_height[i] 
        # Make crop y dims equal to image ydims
        df.ymin[i] = 0
        # Center crop x dims
        df.xmin[i] = df.xmin[i] + 0.5*(crop_w - df.im_height[i]) 
    # Where crop width is greater than crop height, but neither is smaller than image dimensions
    elif crop_w > crop_h and \
    min(crop_w, crop_h) <= min(df.im_height[i], df.im_width[i]):
        # Do not crop, set values equal to image dimensions
        df.crop_width[i] = df.im_width[i]
        df.crop_height[i] = df.im_height[i] 
        df.ymin[i] = 0
        df.xmin[i] = 0  

    # Crop Width == Crop Height
    # See project wiki "Detailed explanation with drawings: convert_bboxdims.py", Scenario 3
    # Where crop width equals crop height and both/either are smaller than image dimensions
    elif crop_w == crop_h and \
    (crop_h + pad) <= min(df.im_height[i], df.im_width[i]):  
        # X and Y dims not OOB (out of bounds)
        if (df.ymax[i] + pad) <= df.im_height[i] and \
        (df.xmax[i] + pad) <= df.im_width[i]:
            pass # do nothing, padding already included in ymin/xmin values
        # X dims OOB (+), shift cropping box left
        elif (df.ymax[i] + pad) > df.im_height[i] and \
        (df.xmax[i] + pad) <= df.im_width[i]:
            df.ymin[i] = df.ymin[i] - (df.ymax[i] + pad - df.im_height[i])
        # Y dims OOB (+), shift cropping box down
        elif (df.ymax[i] + pad) <= df.im_height[i] and \
        (df.xmax[i] + pad) > df.im_width[i]:  
            df.xmin[i] = df.xmin[i] - (df.xmax[i] + pad - df.im_width[i])
        # X and Y dims OOB (+), shift cropping box down and left   
        elif (df.ymax[i] + pad) > df.im_height[i] and \
        (df.xmax[i] + pad) > df.im_width[i]:
            df.ymin[i] = df.ymin[i] - (df.ymax[i] + pad - df.im_height[i])
            df.xmin[i] = df.xmin[i] - (df.xmax[i] + pad - df.im_width[i])
    # Where crop height equals crop width, but neither are within image dimensions
    elif crop_w == crop_h and \
    min(crop_w, crop_h) <= min(df.im_height[i], df.im_width[i]):
        # Do not crop, set values equal to image dimensions
        df.crop_width[i] = df.im_width[i]
        df.crop_height[i] = df.im_height[i] 
        df.ymin[i] = 0
        df.xmin[i] = 0 

## Image coordinates should be positive, set negative xmin and ymin values to 0
df.xmin[df.xmin < 0] = 0
df.ymin[df.ymin < 0] = 0
print(df.head())

# Test that dimensions were modified appropriately for dataset by exporting crop coordinates to display_test.tsv 
# Load this file into crop_coords_display_test.ipynb and visualize results
df.to_csv('object_detection_for_image_cropping/data_files/output/Lepidoptera/lepidoptera_crops_rcnn_20000img_display_test.tsv', sep='\t', index=True)

# Get image and cropping dimensions in EOL format (concatenated string with labels)
# {"height":"423","width":"640","crop_x":123.712,"crop_y":53.4249,"crop_width":352,"crop_height":0}
df['crop_dimensions'] = np.nan
for i, row in df.iterrows():
    df.crop_dimensions[i] = ('{{"height":"{}","width":"{}","crop_x":{},"crop_y":{},"crop_width":{},"crop_height":{}}}'
    .format(df.im_height[i], df.im_width[i], df.xmin[i], df.ymin[i], df.crop_width[i], df.crop_height[i]))
df.reset_index(inplace=True)
print(df.head())

# Create EOL crops formatted dataframe from cols: identifier, dataobjectversionid, eolmediaurl, and crop_dimensions
eol_crops = pd.DataFrame(df.iloc[:,np.r_[-3,-2,0,-1]])
print(eol_crops.head())

# Write results to tsv formmatted to EOL crop coordinate standards
eol_crops.to_csv('object_detection_for_image_cropping/data_files/output/Lepidoptera/lepidoptera_crops_rcnn_20000img.tsv', sep='\t', index=False)

# Print time to run script
print ('Run time: {} seconds'.format(format(time.time()- start, '.2f')))