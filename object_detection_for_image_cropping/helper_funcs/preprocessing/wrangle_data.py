# Utility functions for dataset preprocessing - Classification for image tagging and Object detection for image cropping
# Last updated 21 Aug 2025 by K Wolcott
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
import cv2
import csv

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

# To draw cropping coordinates on an image
def draw_boxes(image, box, class_name):
  image_wboxes = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), \
                               (255, 0, 157), 3) # change box color and thickness

  return image_wboxes

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

# To augment an image with bounding boxes
# modified from https://github.com/aleju/imgaug
def augment_image_w_bboxes(image, crops, i, filter, folder, cwd, display_results=False):
    # Define image augmentation pipeline
    seq = iaa.Sequential([
            iaa.Crop(px=(1, 16), keep_size=False), # crop by 1-16px, resize resulting image to orig dims
            iaa.Affine(rotate=(-25, 25)), # rotate -25 to 25 degrees
            iaa.GaussianBlur(sigma=(0, 3.0)), # blur using gaussian kernel with sigma of 0-3
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True)
            ])
    # Set seed to make augmentation reproducible across runs, otherwise will be random each time
    ia.seed(1)

    # Organize locations to save files and relevant tag to filter by
    pathbase = folder + '/'
    class_name = filter

    # Define image info needed for export
    im_h, im_w = image.shape[:2]
    xmin = crops.xmin[i].astype(int)
    ymin = crops.ymin[i].astype(int)
    xmax = crops.xmax[i].astype(int)
    ymax = crops.ymax[i].astype(int)
    box = [xmin, ymin, xmax, ymax]
    fn = str(crops['data_object_id'][i]) + '.jpg'
    fpath = pathbase + fn

    # Export unaugmented image info for future use training object detectors
    outfpath = cwd + '/' + filter + '_crops_train_aug.tsv'
    print("Saving augmented cropping data to: ", outfpath)
    with open(outfpath, 'a') as out_file:
          tsv_writer = csv.writer(out_file, delimiter='\t')
          tsv_writer.writerow([crops.data_object_id[i], crops.obj_url[i], \
                              im_h, im_w, box[0], box[1], \
                              box[2], box[3], fn, fpath, class_name])

    # Load original bounding box coordinates to imgaug format
    bb  = BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)
    bb = BoundingBoxesOnImage([bb], shape=image.shape)

    # Augment image using settings defined above in seq
    image_aug, bb_aug = seq.augment(image=image, bounding_boxes=bb)

    # Define augmentation results needed for export
    fn_aug = str(crops['data_object_id'][i]) + '_aug' + '.jpg'
    fpath_aug = pathbase + fn_aug
    im_h_aug, im_w_aug = image_aug.shape[:2]
    xmin_aug = bb_aug.bounding_boxes[0].x1.astype(int)
    ymin_aug = bb_aug.bounding_boxes[0].y1.astype(int)
    xmax_aug = bb_aug.bounding_boxes[0].x2.astype(int)
    ymax_aug = bb_aug.bounding_boxes[0].y2.astype(int)
    box_aug = [xmin_aug, ymin_aug, xmax_aug, ymax_aug]

    # Export augmentation results for future use training object detectors
    with open(outfpath, 'a') as out_file:
          tsv_writer = csv.writer(out_file, delimiter='\t')
          tsv_writer.writerow([crops.data_object_id[i], crops.obj_url[i], \
                              im_h_aug, im_w_aug, box_aug[0], box_aug[1], \
                              box_aug[2], box_aug[3], fn_aug, fpath_aug, class_name])

    # Draw augmented bounding box and image
    # Only use for up to 50 images
    if display_results:
        image_wboxes = draw_boxes(image_aug, box_aug, class_name)
        display_image(image_wboxes)
        url = crops["obj_url"][i]
        plt.title('{}) Successfully augmented image from {}'.format(format(i+1, '.0f'), url))

    return image_aug, fpath_aug

# Filter by rating of interest
def filter_by_rating(df, filter=filter, disp_head=False):
    rating = df.loc[round(df["overall_rating"])==int(filter)]
    rating = rating["obj_url"].copy()
    
    if disp_head:
          print("Rating = {}}:\n {}".format(filter, rating.head()))
    print("\n Number of available ratings for training/testing class {}: \n {}".format(filter, len(rating)))

    return rating

# Reformat cropping dimensions
def reformat_crops(crops, disp_head=True):
    # Remove/replace characters in crop_dimensions string
    crops.crop_dimensions.replace('"|{|}', '', regex=True, inplace=True)
    crops.crop_dimensions.replace(':', ',', regex=True, inplace=True)

    # Split crop_dimensions into their own columns
    cols = crops.crop_dimensions.str.split(",", expand=True)
    crops["im_height"] = cols[1]
    crops["im_width"] = cols[3]
    crops["xmin"] = cols[5]
    crops["ymin"] = cols[7]
    crops["xmax"] = cols[5].astype(float) + cols[9].astype(float) # add cropwidth to xmin, note crops are square so width=height
    crops["ymax"] = cols[7].astype(float) + cols[9].astype(float) # add cropheight to ymin, note crops are square so width=height

    # Remove crop_dimensions column
    crops.drop(columns =["crop_dimensions"], inplace = True)
    if disp_head:
        print("\n~~~Reformatted EOL crops head~~~\n", crops.head())

    return crops

# Filter by taxon of interest
def filter_by_taxon(crops, filter=filter, disp_head=False):
    taxon = crops.loc[crops.ancestry.str.contains(filter, case=False, na=False)]
    taxon.drop(columns =["ancestry"], inplace = True)
    taxon['name'] = filter
    taxon.reset_index(inplace=True)
    if disp_head:
          print("Showing dataset for only {}: {}\n".format(filter, taxon.head()))
    print("\n~~~Number of available cropping coordinates for training/testing with {}~~~: \n{}\n".format(filter, len(taxon)))

    return taxon

# Split into train and test datasets
def split_train_test(crops, outfpath, frac, disp_head=False):
    # Randomly select 80% of data to use for training (set random_state seed for reproducibility)
    idx = crops.sample(frac = 0.8, random_state=2).index
    train = crops.iloc[idx]
    if disp_head:
        print("Training data for {} (n={} crops): \n".format(filter, len(train), train.head()))

    # Select the remaining 20% of data for testing
    # Uses the inverse index from above
    test = crops.iloc[crops.index.difference(idx)]
    if disp_head:
        print("Testing data for {} (n={} crops): \n".format(filter, len(test), test.head()))

    # Write test and train to tsvs
    train_outfpath = os.path.splitext(outfpath)[0] + '_train' + '.tsv'
    train.to_csv(train_outfpath, sep='\t', header=True, index=False)
    test_outfpath = os.path.splitext(outfpath)[0] + '_test' + '.tsv'
    test.to_csv(test_outfpath, sep='\t', header=True, index=False)
    print("\n Train and test datasets sucessfully split and saved to: \n\n{}\n{}"\
          .format(train_outfpath, test_outfpath))

    return train, test

# Remove out of bounds values
def remove_oob(crops):
    # Set negative values to 0
    crops.loc[crops.xmin < 0, 'xmin'] = 0
    crops.loc[crops.ymin < 0, 'ymin'] = 0

    # Remove out of bounds cropping dimensions
    ## When crop height > image height, set crop height equal to image height
    idx = crops.index[crops.ymax > crops.im_height]
    crops.loc[idx, 'ymin'] = 0
    crops.loc[idx, 'ymax'] = crops.loc[idx, 'im_height']
  
    ## When crop width > image width, set crop width equal to image width
    idx = crops.index[crops.xmax > crops.im_width]
    crops.loc[idx, 'xmin'] = 0
    crops.loc[idx, 'xmax'] = crops.loc[idx, 'im_width']

    # Write relevant results to csv formatted for training and annotations needed by Tensorflow and YOLO
    crops_oobrem = crops[['xmin', 'ymin', 'xmax', 'ymax',
                  'filename', 'im_width', 'im_height', 'class']]

    return crops_oobrem

# Get info from EOL user generated cropping file
def get_image_info(image, crops, i, cwd, folder, filter):
    pathbase = folder + '/'
    class_name = filter

    # Define image info needed for export
    im_h, im_w = image.shape[:2]
    xmin = crops.xmin[i].astype(int)
    ymin = crops.ymin[i].astype(int)
    xmax = crops.xmax[i].astype(int)
    ymax = crops.ymax[i].astype(int)
    box = [xmin, ymin, xmax, ymax]
    fn = str(crops['data_object_id'][i]) + '.jpg'
    fpath = pathbase + fn

    # Export to crops_test.tsv
    fpath = cwd + "/" + filter + "_crops_test.tsv"
    outfpath = os.path.splitext(fpath)[0] + '_notaug.tsv'
    with open(outfpath, 'a') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow([crops.data_object_id[i], crops.obj_url[i], \
                                 im_h, im_w, box[0], box[1], box[2], box[3], \
                                 fn, fpath, class_name])

    return fpath

# Draw cropping box on image
def draw_box_on_image(df, img, i):
    # Get box coordinates
    xmin = df['xmin'][i].astype(int)
    ymin = df['ymin'][i].astype(int)
    xmax = df['xmax'][i].astype(int)
    ymax = df['ymax'][i].astype(int)
    box = [xmin, ymin, xmax, ymax]

    # Set box/font color and size
    maxdim = max(df['im_height'][i],df['im_width'][i])
    fontScale = maxdim/600
    box_col = (255, 0, 157)

    # Add label to image
    tag = df['class'][i]
    image_wbox = cv2.putText(img, tag, (xmin+7, ymax-12), cv2.FONT_HERSHEY_SIMPLEX, fontScale, box_col, 2, cv2.LINE_AA)

    # Draw box label on image
    image_wbox = cv2.rectangle(img, (xmin, ymax), (xmax, ymin), box_col, 5)

    return image_wbox, box
