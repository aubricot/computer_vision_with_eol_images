# Utility functions for running inference
# Last updated 14 Jan 2025 by K Wolcott

# For downloading and displaying images
import matplotlib
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For running inference
import tensorflow as tf

# For working with data
import pandas as pd
import numpy as np
# Suppress pandas warning about writing over a copy of data
pd.options.mode.chained_assignment = None  # default='warn'
from functools import reduce
from urllib.error import HTTPError
# So URL's don't get truncated in display
pd.set_option('display.max_colwidth',1000)
pd.options.display.max_columns = None

# For drawing onto images
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring inference time
import time

# To read in EOL formatted data files
def read_datafile(fpath, sep="\t", header=0, disp_head=True, lineterminator='\n', encoding='latin1', dtype=None):
    try:
        df = pd.read_csv(fpath, sep=sep, header=header, lineterminator=lineterminator, encoding=encoding, dtype=dtype)
        if disp_head:
          print("Data header: \n", df.head())
    except FileNotFoundError as e:
        raise Exception("File not found: Enter the path to your file in form field and re-run").with_traceback(e.__traceback__)

    return df

# To display loaded image
def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)

# To load image in and do something with it
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

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

# For reading in images from URL and passing through TF models for inference
# Modified from TF Hub https://www.tensorflow.org/hub/tutorials/object_detection
def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    im_h, im_w = pil_image.size
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.LANCZOS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    #print("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename, im_h, im_w

# To draw bounding boxes on an image
# Modified from TF Hub https://www.tensorflow.org/hub/tutorials/object_detection
def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax,
                               color, font, thickness=4, display_str_list=()):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                 ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
              (left, top)],
              width=thickness,
              fill=color)

    # Adjust display string placement if out of bounds
    display_str_heights = [font.getbbox(ds)[3]-font.getbbox(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for ds in display_str_list[::-1]:
        text_height = font.getbbox(ds)[3] - font.getbbox(ds)[1]
        text_width = font.getbbox(ds)[2] - font.getbbox(ds)[0]
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  ds, fill="black", font=font)
        text_bottom -= text_height - 2 * margin

# Filter detections and annotate images with results 
# Modified from TF Hub https://www.tensorflow.org/hub/tutorials/object_detection
def draw_boxes(image, boxes, class_names, scores, max_boxes, min_score, filter, label_map):
    # Format text above boxes
    colors = list(ImageColor.colormap.values())
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    # Draw up to N-max boxes with confidence > score threshold
    for i in range(0, max_boxes):
        if scores[0][i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[0][i])
            display_str = "{}: {}%".format(label_map[class_names[0][i]],
                                     int(100 * scores[0][i]))
            color = colors[hash(class_names[0][i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            # Only the filtered class is shown on images
            if filter in display_str:
                draw_bounding_box_on_image(
                    image_pil,
                    ymin, xmin, ymax, xmax,
                    color, font, display_str_list=[display_str])
                np.copyto(image, np.array(image_pil))
    return image

# For running inference
# Modified from TF Hub https://www.tensorflow.org/hub/tutorials/object_detection
def run_detector_tf(detector, image_url, outfpath, filter, label_map, max_boxes, min_score):
    image_path, im_h, im_w = download_and_resize_image(image_url, 640, 480)
    img = load_img(image_path)
    converted_img  = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]

    # Actual detection
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key:value.numpy() for key,value in result.items()}
    print("Found %d objects with > %s confidence" % (min(result["num_detections"], max_boxes), min_score))
    print("Inference time: %s sec" % format(end_time-start_time, '.2f'))

    # Draw detection boxes on image
    image_wboxes = draw_boxes(img.numpy(), result["detection_boxes"],
                                  result["detection_classes"], result["detection_scores"], max_boxes, min_score, filter, label_map)

    return image_wboxes, result, im_h, im_w

# Convert normalized detection coordinates (scaled to 0,1) to pixel values
def denormalize_coords(crops):
    crops.xmin = crops.xmin * crops.im_width
    crops.ymin = crops.ymin * crops.im_height
    crops.xmax = crops.xmax * crops.im_width
    crops.ymax = crops.ymax * crops.im_height
    # Round results to 2 decimal places
    crops.round(2)
    #print("De-normalized cropping coordinates: \n", crops.head())

    return crops

# For images with >1 detection, make a 'super box' that containings all boxes
def make_superboxes(crops):
    # Get superbox coordinates that contain all detection boxes per image
    xmin = pd.DataFrame(crops.groupby(['url'])['xmin'].min()) # smallest xmin
    ymin = pd.DataFrame(crops.groupby(['url'])['ymin'].min()) # smallest ymin
    xmax = pd.DataFrame(crops.groupby(['url'])['xmax'].max()) # largest xmax
    ymax = pd.DataFrame(crops.groupby(['url'])['ymax'].max()) # largest ymax

    # Workaround to get im_height, im_width and class in same format as 'super box' coords
    # There is only one value for im_height and im_width, so taking max will return unchanged values
    im_h = pd.DataFrame(crops.groupby(['url'])['im_height'].max())
    im_w = pd.DataFrame(crops.groupby(['url'])['im_width'].max())
    im_class = pd.DataFrame(crops.groupby(['url'])['class_name'].max())

    # Make list of superboxes
    superbox_list = [im_h, im_w, xmin, ymin, xmax, ymax, im_class]

    # Make a new dataframe with 1 superbox per image
    superbox_df = reduce(lambda  left, right: pd.merge(left, right, on=['url'],
                                            how='outer'), superbox_list)
    #print("Cropping dataframe, 1 superbox per image: \n", crops_unq.head())

    return superbox_df

# Add EOL img identifying info from breakdown file to cropping data
def add_identifiers(*, bundle_info, crops):
    # Get dataObjectVersionIDs, identifiers, and eolMediaURLS from indexed cols
    ids = bundle_info.iloc[:, np.r_[0:2,-2]]
    ids.set_index('eolMediaURL', inplace=True, drop=True)
    #print("Bundle identifying info head: \n", ids.head())

    # Set up superboxes df for mapping to bundle_info
    superboxes.reset_index(inplace=True)
    superboxes.rename(columns={'url': 'eolMediaURL'}, inplace=True)
    superboxes.set_index('eolMediaURL', inplace=True, drop=True)

    # Map dataObjectVersionIDs to crops_unq using eolMediaURL as the index
    crops_w_identifiers = pd.DataFrame(superboxes.merge(ids, left_index=True, right_index=True))
    crops_w_identifiers.reset_index(inplace=True)
    print("\n Crops with added EOL identifiers: \n", crops_w_identifiers.head())

    return crops_w_identifiers

# Check if dimensions are out of bounds
def are_dims_oob(dim):
    # Check if square dimensions are out of image bounds (OOB)
    if dim > min(im_h, im_w):
        return True
    else:
        return False

# Center padded, square coordinates around object midpoint
def center_coords(coord_a, coord_b, crop_w, crop_h, im_dim_a, im_dim_b, pad):
    # Centered, padded top-right coordinates
    tr_coord_a = coord_a + 0.5*(abs(crop_h - crop_w)) + pad
    tr_coord_b = coord_a + pad
    # Adjust coordinate positions if OOB (out of bounds)
    if crop_h != crop_w: # for cond 1 and 2
        # Both coords not OOB
        if (tr_coord_a <= im_dim_a) and (tr_coord_b <= im_dim_b):
            bl_coord_a = coord_a - 0.5*(abs(crop_h - crop_w)) - pad
            bl_coord_b = coord_b - pad
        # Topright coord_a OOB (+), shift cropping box down/left a-axis
        elif (tr_coord_a > im_dim_a) and (tr_coord_b <= im_dim_b):
            bl_coord_a = 0.5*(abs(im_dim_a - crop_w))
            bl_coord_b = coord_b - pad
        # Topright coord_b OOB (+), shift cropping box down/left b-axis
        elif (tr_coord_a <= im_dim_a) and (tr_coord_b > im_dim_b):
            bl_coord_a = coord_a - 0.5*(abs(crop_h - crop_w)) - pad
            bl_coord_b = coord_b - (tr_coord_b - im_dim_b + pad)
        # Both coords OOB (+), shift cropping box down/left both axes
        elif (tr_coord_a > im_dim_a) and (tr_coord_b > im_dim_b):
            bl_coord_a = 0.5*(abs(im_dim_a - crop_w))
            bl_coord_b = coord_b - (tr_coord_b - im_dim_b + pad)
    else: # for cond 3
        # Both coords not OOB
        if (tr_coord_a <= im_dim_a) and (tr_coord_b <= im_dim_b):
            bl_coord_a = coord_a - pad
            bl_coord_b = coord_b - pad
        # Topright coord_a OOB (+), shift cropping box down/left a-axis
        elif (tr_coord_a > im_dim_a) and (tr_coord_b <= im_dim_b):
            bl_coord_a = coord_a - (tr_coord_a - im_dim_a + pad)
            bl_coord_b = coord_b - pad
        # Topright coord_b OOB (+), shift cropping box down/left b-axis
        elif (tr_coord_a <= im_dim_a) and (tr_coord_b > im_dim_b):
            bl_coord_a = coord_a - pad
            bl_coord_b = coord_b - (tr_coord_b - im_dim_b + pad)
        # Both coords OOB (+), shift cropping box down/left both axes
        elif (tr_coord_a > im_dim_a) and (tr_coord_b > im_dim_b):
            bl_coord_a = coord_a - (tr_coord_a - im_dim_a + pad)
            bl_coord_b = coord_b - (tr_coord_b - im_dim_b + pad)

    return bl_coord_a, bl_coord_b

# Set square dimensions = larger bounding box side
def make_large_square(dim):
    # Set new square crop dims = original larger crop dim
    lg_square = crop_w1 = crop_h1 = dim
    return lg_square

# Set square dimensions = smaller bounding box side
def make_small_square(dim):
    # Set new square crop dims = original smaller crop dim
    sm_square = crop_w1 = crop_h1 = dim
    return sm_square

# Add x% padding to bounding box dimensions
def add_padding(dim):
    # Add padding on all sides of square
    padded_dim = dim + 2*percent_pad*dim
    return padded_dim