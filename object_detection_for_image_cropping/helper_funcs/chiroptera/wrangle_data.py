# Utility functions for running inference - Chiroptera
# Last updated 19 Jan 2025 by K Wolcott

# For downloading and displaying images
import matplotlib
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
import urllib
from six import BytesIO
import cv2

# For running inference
import tensorflow as tf

# For working with data
import pandas as pd
import numpy as np
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
    plt.show()

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
def draw_boxes(image, boxes, class_names, scores, max_boxes, min_score, filter, label_map, category_index):
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
            display_str = "{}: {}%".format(category_index[class_names[0][i]]['name'],
                                     int(100 * scores[0][i]))
            color = colors[hash(class_names[0][i]) % len(colors)]
            image_pil = Image.fromarray(np.squeeze(image))
            # Only the filtered class is shown on images
            if filter in display_str:
                draw_bounding_box_on_image(
                    image_pil,
                    ymin, xmin, ymax, xmax,
                    color, font, display_str_list=[display_str])
                np.copyto(image, np.array(image_pil))
    return image[0]

# For running inference
# Modified from TF Hub https://www.tensorflow.org/hub/tutorials/object_detection
def run_detector_tf(detection_graph, image_url, outfpath, filter, label_map, max_boxes, min_score, category_index):
    image_np, im_h, im_w = url_to_image(image_url)
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection
            start_time = time.time()
            result = sess.run([detection_boxes, detection_scores,
                               detection_classes, num_detections],
                               feed_dict={image_tensor: image_np})
            end_time = time.time()

            result = {"detection_boxes": result[0], "detection_scores": result[1],
                      "detection_classes": result[2], "num_detections": result[3]}

            print("Found %d objects with > %s confidence" % (min(result["num_detections"], max_boxes), min_score))
            print("Inference time: %s sec" % format(end_time-start_time, '.2f'))

            # Draw detection boxes on image
            image_wboxes = draw_boxes(image_np, result["detection_boxes"],
                                      result["detection_classes"], result["detection_scores"], 
                                      max_boxes, min_score, filter, label_map, category_index)

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
def add_identifiers(superboxes, bundle_info):
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

# For uploading an image from url
# Modified from https://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv/
def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np = np.expand_dims(image, axis=0)
    im_h, im_w = image.shape[:2]
  
    return image_np, im_h, im_w

# Draw cropping box on image
def draw_box_on_image(df, i, img):
    # Get box coordinates
    xmin = df['xmin'][i].astype(int)
    ymin = df['ymin'][i].astype(int)
    xmax = df['xmax'][i].astype(int)
    ymax = df['ymax'][i].astype(int)
    boxcoords = [xmin, ymin, xmax, ymax]

    # Set box/font color and size
    maxdim = max(df['im_height'][i],df['im_width'][i])
    fontScale = maxdim/600
    box_col = (255, 0, 157)

    # Add label to image
    tag = df['class_name'][i]
    img = np.squeeze(img)
    image_wbox = cv2.putText(img, tag, (xmin+7, ymax-12), cv2.FONT_HERSHEY_SIMPLEX, fontScale, box_col, 2, cv2.LINE_AA)

    # Draw box label on image
    image_wbox = cv2.rectangle(img, (xmin, ymax), (xmax, ymin), box_col, 5)

    return image_wbox, boxcoords

# Convert rectangles to square crops that are within image bounds (Optional: add padding)
def make_square_crops(df, pad=0):
    print("Before making square: \n", df.head())
    start_time = time.time()
    df['crop_height'] = None
    df['crop_width'] = None
    for i, row in df.iterrows():
        # Calculate original (rectangular) crop bounding box dimensions
        crop_h0 = round(df['ymax'][i] - df['ymin'][i], 1)
        crop_w0 = round(df['xmax'][i] - df['xmin'][i], 1)
        # Define image dimensions
        im_h = df.im_height[i]
        im_w = df.im_width[i]
        # Define original (rectangular) crop bounding box coordinates
        xmin0 = df.xmin[i]
        ymin0 = df.ymin[i]
        xmax0 = df.xmax[i]
        ymax0 = df.ymax[i]

        # Calculate the center of the crop bounding box
        center_x = (xmin0 + xmax0) // 2
        center_y = (ymin0 + ymax0) // 2

        # Make crop square using the longer bounding box side length
        side = max(crop_h0, crop_w0) + 2 * pad

        # Ensure the square stays within image bounds
        df.loc[i, 'xmin'] = xmin1 = max(0, center_x - side // 2)
        df.loc[i, 'ymin'] = ymin1 = max(0, center_y - side // 2)
        xmax1 = min(im_w, center_x + side // 2)
        ymax1 = min(im_h, center_y + side // 2)

        # Define new crop width and height that are within bounds
        crop_h1 = round(ymax1 - ymin1, 1)
        crop_w1 = round(xmax1 - xmin1, 1)
        crop_final = min(crop_h1, crop_w1)
        df.loc[i, 'crop_height'] = crop_final
        df.loc[i, 'crop_width'] = crop_final
        df.loc[i, 'xmax'] = xmin1 + crop_final
        df.loc[i, 'ymax'] = ymin1 + crop_final

    # Print progress message
    print("Cropping coordinates, made square and with {}% padding: \n{}".format(pad*100, df.head()))
    # Print time to run script
    print('Run time: {} seconds'.format(format(time.time()- start_time, '.2f')))

    return df