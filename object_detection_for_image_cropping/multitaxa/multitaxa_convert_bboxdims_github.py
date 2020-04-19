# Converting object detection bounding box coordinates to EOL crop format
# Last modified 15 April 20

import time
start = time.time()
import csv
import numpy as np
import pandas as pd
import os

# Read in crop file exported from merge_tsvs.py (after export from multitaxa_train_tf_rcnns.ipynb)
# TO DO: Read in and process crop data separately for each taxon
crops = pd.read_csv('object_detection_for_image_cropping/data_files/input/Multitaxa/squamata_det_crops_20000.tsv', sep='\t', header=0)
#crops = pd.read_csv('object_detection_for_image_cropping/data_files/input/Multitaxa/coleoptera_det_crops_20000.tsv', sep='\t', header=0)
#crops = pd.read_csv('object_detection_for_image_cropping/data_files/input/Multitaxa/anura_det_crops_20000.tsv', sep='\t', header=0)
#crops = pd.read_csv('object_detection_for_image_cropping/data_files/input/Multitaxa/carnivora_det_crops_20000.tsv', sep='\t', header=0)
print(crops.head())

# Correct for images with 1+ bounding boxes by making a 'super box' containing all small boxes per image
# For each image, take smallest xmin, ymin and largest xmax, ymax from bbox coordinates
xmin_unq = pd.DataFrame(crops.groupby(['image_url'])['xmin'].min())
ymin_unq = pd.DataFrame(crops.groupby(['image_url'])['ymin'].min())
xmax_unq = pd.DataFrame(crops.groupby(['image_url'])['xmax'].max())
ymax_unq = pd.DataFrame(crops.groupby(['image_url'])['ymax'].max())
print(xmin_unq.head())

# Workaround to get im_height, im_width and class in same format/order as 'super box' coords
# There is only one value for im_height and im_width, so taking max will return unchanged values
im_height_unq = pd.DataFrame(crops.groupby(['image_url'])['im_height'].max())
im_width_unq = pd.DataFrame(crops.groupby(['image_url'])['im_width'].max())
taxon_unq = pd.DataFrame(crops.groupby(['image_url'])['class'].max())

# Make a new dataframe with the unique values (ie. for only 1 bbox/image)
crops_unq = im_height_unq.merge(im_width_unq, left_index=True, right_index=True)
crops_unq = crops_unq.merge(taxon_unq, left_index=True, right_index=True)
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
#bd = pd.read_csv('object_detection_for_image_cropping/data_files/input/Lepidoptera/images_for_Lepidoptera_20K_breakdown_000001.txt', sep='\t', header=0)
# Combine EOL image bundles for all taxa
all_filenames = ["images_for_Squamata_20K_breakdown_000001.txt", "images_for_Coleoptera_20K_breakdown_000001.txt", "images_for_Anura_20K_breakdown_000001.txt", "images_for_Carnivora_20K_breakdown_000001.txt"]
bd = pd.concat([pd.read_csv(f, sep='\t', header=0) for f in all_filenames], ignore_index=True, sort=False)
bd = bd.iloc[:, np.r_[0:2,-3]]
print(bd.head())

## Map dataObjectVersionIDs to crops_unq using identifiers as the index
crops_unq.set_index('eolMediaURL', inplace=True, drop=True)
bd.set_index('eolMediaURL', inplace=True, drop=True)
df = crops_unq.merge(bd, left_index=True, right_index=True)
print(df.head())

# Convert bounding box/cropping dimensions to square, add padding, and make sure crop boxes aren't out of image bounds
for i, row in df.iterrows():
    # Optional: Pad by xx% larger crop dimension (height)
    # Note: 0% pad chosen for Coleoptera, Anura, Carnivora; X% for Squamata
    pad = 0 * max(df.crop_height[i], df.crop_width[i])
    # Define variables for use filtering data through loops below
    crop_h0 = df.crop_height[i]
    crop_w0 = df.crop_width[i]
    im_h = df.im_height[i]
    im_w = df.im_width[i]
    xmin0 = df.xmin[i]
    ymin0 = df.ymin[i]
    xmax0 = df.xmax[i]
    ymax0 = df.ymax[i]

    # Crop Height > Crop Width
    # See project wiki "Detailed explanation with drawings: convert_bboxdims.py", Scenario 1
    if crop_h0 > crop_w0:
        # Where padded crop height is within image dimensions
        if (crop_h0 + 2*pad) <= min(im_h, im_w):   
            # Make new crop dimensions square by setting crop height equal to crop width
            df.crop_height[i] = crop_h1 = crop_h0 + 2*pad  
            df.crop_width[i] = crop_w1 = df.crop_height[i]
            # X and Y dims not OOB (out of bounds)
            if (xmax0 + 0.5*(crop_h0 - crop_w0) + pad) <= im_w and \
            (ymax0 + pad) <= im_h:
                df.xmin[i] = xmin0 - 0.5*(crop_h0 - crop_w0) - pad
                df.ymin[i] = ymin0 - pad
            # X dims OOB (+), shift cropping box left
            elif (xmax0 + 0.5*(crop_h0 - crop_w0) + pad) > im_w and \
            (ymax0 + pad) <= im_h:
                df.xmin[i] = 0.5*(im_w - crop_w1)
                df.ymin[i] = ymin0 - pad
            # Y dims OOB (+), shift cropping box down
            elif (xmax0 + 0.5*(crop_h0 - crop_w0) + pad) <= im_w and \
            (ymax0 + pad) > im_h:  
                df.xmin[i] = xmin0 - 0.5*(crop_h0 - crop_w0) - pad
                df.ymin[i] = ymin0 - (ymax0 + pad - im_h)
            # X and Y dims OOB (+), shift cropping box down and left   
            elif (xmax0 + 0.5*(crop_h0 - crop_w0) + pad) > im_w and \
            (ymax0 + pad) > im_h:
                df.xmin[i] = 0.5*(im_w - crop_w1)
                df.ymin[i] = ymin0 - (ymax0 + pad - im_h)  
        # Where padded crop height is not within image dimensions, but un-padded is
        elif (crop_h0 + 2*pad) > min(im_h, im_w) and \
        crop_h0 <= min(im_h, im_w):
            # Make new crop dimensions square by setting crop height equal to crop width
            df.crop_height[i] = crop_h1 = min(im_h, im_w)
            df.crop_width[i] = crop_w1 = df.crop_height[i]   
            # Center cropping coordinates
            df.xmin[i] = xmin0 - 0.5*(min(im_h, im_w) - crop_w0)
            df.ymin[i] = 0
        # Where crop height is not within image dimensions, but padded crop width is
        elif crop_h0 > min(im_h, im_w) and \
            (crop_w0 + 2*pad) <= min(im_h, im_w):    
            # Make new crop dimensions square by setting crop height equal to crop width
            df.crop_width[i] = crop_w1 = crop_w0 + 2*pad
            df.crop_height[i] = crop_h1 = df.crop_width[i]  
            # Center crop dimensions
            df.xmin[i] = xmin0 - pad
            df.ymin[i] = ymin0 + 0.5*(crop_h0 - crop_w0) - pad   
        # Where crop height is not within image dimensions, but un-padded crop width is
        elif crop_h0 > min(im_h, im_w) and \
            (crop_w0 + 2*pad) > min(im_h, im_w) and \
            crop_w0 <= min(im_h, im_w):    
            # Make new crop dimensions square by setting crop height equal to image width
            df.crop_width[i] = crop_w1 = min(im_h, im_w)
            df.crop_height[i] = crop_h1 = df.crop_width[i]  
            # Center crop dimensions (note that min(im_w, im_h) is cancelled out b/c was + and - to get new ymin) 
            df.ymin[i] = ymin0 + 0.5*(crop_h0 - crop_w0 - pad)     
            df.xmin[i] = 0          
        # Where crop height is greater than width, but neither is within than image dimensions
        elif min(crop_h0, crop_w0) > min(im_h, im_w):
            # Do not crop, set values equal to image dimensions
            df.crop_width[i] = crop_w1 = im_w
            df.crop_height[i] = crop_h1 = im_h 
            df.ymin[i] = 0
            df.xmin[i] = 0  
    
    # Crop Width > Crop Height
    # See project wiki "Detailed explanation with drawings: convert_bboxdims.py", Scenario 2
    elif crop_w0 > crop_h0:
        # Where padded crop width is within image dimensions 
        if (crop_w0 + 2*pad) <= min(im_h, im_w):    
            # Make new crop dimensions square by setting crop height equal to crop width
            df.crop_width[i] = crop_w1 = crop_w0 + 2*pad  
            df.crop_height[i] = crop_h1 = df.crop_width[i] 
            # X and Y dims not OOB (out of bounds)
            if (ymax0 + 0.5*(crop_w0 - crop_h0) + pad) <= im_h and \
            (xmax0 + pad) <= im_w:
                df.ymin[i] = ymin0 - 0.5*(crop_w0 - crop_h0) - pad
                df.xmin[i] = xmin0 - pad
            # X dims OOB (+), shift cropping box left
            elif (ymax0 + 0.5*(crop_w0 - crop_h0) + pad) <= im_h and \
            (xmax0 + pad) > im_w:  
                df.ymin[i] = ymin0 - 0.5*(crop_w0 - crop_h0) - pad
                df.xmin[i] = xmin0 - (xmax0 + pad - im_w)
            # Y dims OOB (+), shift cropping box down
            elif (ymax0 + 0.5*(crop_w0 - crop_h0) + pad) > im_h and \
            (xmax0 + pad) <= im_w:
                df.ymin[i] = 0.5*(im_h - crop_h1)
                df.xmin[i] = xmin0 - pad
            # X and Y dims OOB (+), shift cropping box down and left   
            elif (ymax0 + 0.5*(crop_w0 - crop_h0) + pad) > im_h and \
            (xmax0 + pad) > im_w:
                df.ymin[i] = 0.5*(im_h - crop_h1)
                df.xmin[i] = xmin0 - (xmax0 + pad - im_w)
        # Where padded crop width is not within image dimensions, but un-padded is
        elif (crop_w0 + 2*pad) > min(im_h, im_w) and \
        crop_w0 <= min(im_h, im_w):    
            # Make new crop dimensions square by setting crop height equal to image width
            df.crop_width[i] = crop_w1 = min(im_h, im_w)
            df.crop_height[i] = crop_h1 = df.crop_width[i]
            # Center cropping coordinates
            df.ymin[i] = ymin0 - 0.5*(min(im_h, im_w) - crop_h0)
            df.ymin[i] = 0
        # Where crop width is not within image dimensions, but padded crop height is
        elif crop_w0 > min(im_h, im_w) and \
        (crop_h0 + 2*pad) <= min(im_h, im_w):           
            # Make new crop dimensions square by setting crop height equal to crop width
            df.crop_height[i] = crop_h1 = crop_h0 + 2*pad
            df.crop_width[i] = crop_w1 = df.crop_height[i] 
            # Center crop dimensions
            df.ymin[i] = ymin0 - pad
            df.xmin[i] = xmin0 + 0.5*(crop_w0 - crop_h0) - pad 
        # Where crop width is not within image dimensions, but un-padded crop height is
        elif crop_w0 > min(im_h, im_w) and \
        (crop_h0 + 2*pad) > min(im_h, im_w) and \
        crop_h0 <= min(im_h, im_w):           
            # Make new crop dimensions square by setting crop height equal to image width
            df.crop_height[i] = crop_h1 = min(im_h, im_w)
            df.crop_width[i] = crop_w1 = df.crop_height[i] 
            # Center crop dimensions
            df.xmin[i] = xmin0 + 0.5*(crop_w0 - crop_h0 - pad)
            df.ymin[i] = 0
        # Where crop width is greater than height, but neither is within than image dimensions
        elif min(crop_w0, crop_h0) > min(im_h, im_w):
            # Do not crop, set values equal to image dimensions
            df.crop_width[i] = crop_w1 = im_w
            df.crop_height[i] = crop_h1 = im_h 
            df.ymin[i] = 0
            df.xmin[i] = 0  

    # Crop Width == Crop Height
    # See project wiki "Detailed explanation with drawings: convert_bboxdims.py", Scenario 3
    elif crop_w0 == crop_h0: 
        # Where padded crop height/width is within image dimensions
        if (crop_h0 + 2*pad) <= min(im_h, im_w):
            # Make new crop dimensions square by setting crop height equal to crop width 
            df.crop_height[i] = crop_h1 = crop_h0 + 2*pad
            df.crop_width[i] = crop_w1 = df.crop_height[i] 
            # X and Y dims not OOB (out of bounds)
            if (ymax0 + pad) <= im_h and \
            (xmax0 + pad) <= im_w:
                df.ymin[i] = ymin0 - pad
                df.xmin[i] = xmin0 - pad
            # X dims OOB (+), shift cropping box left
            elif (ymax0 + pad) <= im_h and \
            (xmax0 + pad) > im_w:  
                df.xmin[i] = xmin0 - (xmax0 + pad - im_w)
                df.ymin[i] = ymin0 - pad
            # Y dims OOB (+), shift cropping box down
            elif (ymax0 + pad) > im_h and \
            (xmax0 + pad) <= im_w:
                df.ymin[i] = ymin0 - (ymax0 + pad - im_h)
                df.xmin[i] = xmin0 - pad
            # X and Y dims OOB (+), shift cropping box down and left   
            elif (ymax0 + pad) > im_h and \
            (xmax0 + pad) > im_w:
                df.ymin[i] = ymin0 - (ymax0 + pad - im_h)
                df.xmin[i] = xmin0 - (xmax0 + pad - im_w)
        # Where padded crop height/width is not within image dimensions, but un-padded is
        elif (crop_h0 + 2*pad) > min(im_h, im_w) and \
        crop_h0 <= min(im_h, im_w):
            # Make new crop dimensions square by setting crop height equal to smaller image dim 
            df.crop_width[i] = crop_w1 = min(im_h, im_w) 
            df.crop_height[i] = crop_h1 = min(im_h, im_w) 
            # X and Y dims not OOB (out of bounds)
            if ymax0 <= im_h and \
            xmax0 <= im_w:
                pass
            # X dims OOB (+), shift cropping box left
            elif ymax0 <= im_h and \
            xmax0 > im_w:  
                df.xmin[i] = xmin0 - (xmax0 - im_w)
            # Y dims OOB (+), shift cropping box down
            elif ymax0 > im_h and \
            xmax0 <= im_w:
                df.ymin[i] = ymin0 - (ymax0 - im_h)
            # X and Y dims OOB (+), shift cropping box down and left   
            elif ymax0 > im_h and \
            xmax0 > im_w:
                df.ymin[i] = ymin0 - (ymax0 - im_h)
                df.xmin[i] = xmin0 - (xmax0 - im_w)

## Image coordinates should be positive, set negative xmin and ymin values to 0
df.xmin[df.xmin < 0] = 0
df.ymin[df.ymin < 0] = 0
print(df.head())

# Test that dimensions were modified appropriately for dataset by exporting crop coordinates to display_test.tsv 
# Load this file into crop_coords_display_test.ipynb and visualize results
# TO DO: Export results separately for each taxon
df.to_csv('object_detection_for_image_cropping/data_files/output/Multitaxa/squamata_crops_rcnn_i_20000img_display_test.tsv', sep='\t', index=True)
#df.to_csv('object_detection_for_image_cropping/data_files/output/Multitaxa/coleoptera_crops_rcnn_i_20000img_display_test.tsv', sep='\t', index=True)
#df.to_csv('object_detection_for_image_cropping/data_files/output/Multitaxa/anura_crops_rcnn_i_20000img_display_test.tsv', sep='\t', index=True)
#df.to_csv('object_detection_for_image_cropping/data_files/output/Multitaxa/carnivora_crops_rcnn_i_20000img_display_test.tsv', sep='\t', index=True)

# Get image and cropping dimensions in EOL format (concatenated string with labels)
# {"height":"423","width":"640","crop_x":123.712,"crop_y":53.4249,"crop_width":352,"crop_height":0}
df['crop_dimensions'] = np.nan
for i, row in df.iterrows():
    df.crop_dimensions[i] = ('{{"height":"{}","width":"{}","crop_x":{},"crop_y":{},"crop_width":{},"crop_height":{}}}'
    .format(df.im_height[i], df.im_width[i], df.xmin[i], df.ymin[i], df.crop_width[i], df.crop_height[i]))
df.reset_index(inplace=True)
print(df.head())

# Create EOL crops formatted dataframe from cols: identifier, dataobjectversionid, eolmediaurl, crop_dimensions, and class
eol_crops = pd.DataFrame(df.iloc[:,np.r_[-3,-2,0,-1]])
print(eol_crops.head())

# Write results to tsv formmatted to EOL crop coordinate standards
# TO DO: Export results separately for each taxon
eol_crops.to_csv('object_detection_for_image_cropping/data_files/output/Multitaxa/squamata_crops_rcnn_20000img.tsv', sep='\t', index=False)
#eol_crops.to_csv('object_detection_for_image_cropping/data_files/output/Multitaxa/coleoptera_crops_rcnn_20000img.tsv', sep='\t', index=False)
#eol_crops.to_csv('object_detection_for_image_cropping/data_files/output/Multitaxa/anura_crops_rcnn_20000img.tsv', sep='\t', index=False)
#eol_crops.to_csv('object_detection_for_image_cropping/data_files/output/Multitaxa/carivora_crops_rcnn_20000img.tsv', sep='\t', index=False)

# Print time to run script
print ('Run time: {} seconds'.format(format(time.time()- start, '.2f')))