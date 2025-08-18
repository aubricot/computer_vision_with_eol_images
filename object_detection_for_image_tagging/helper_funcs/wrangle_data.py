# Define functions
import os
import requests
import cv2
import matplotlib
import matplotlib.pyplot as plt
import six.moves.urllib as urllib
import pandas as pd
import numpy as np
# Display full URLs in outputs so you can click them and inspect images
pd.set_option('display.max_colwidth', None)

# For uploading an image from url
def image_from_url(image_url, cwd):
    os.chdir(cwd)
    os.chdir("data/imgs")
    image_filename = image_url.rsplit('/', 1)[-1]
    img_fpath = 'data/imgs/' + image_filename
    img_data = requests.get(image_url).content
    with open(image_filename, 'wb') as handler:
        handler.write(img_data)
    print("\033[92m Successfully downloaded image from URL\033[0m")
    os.chdir(cwd)

    return img_fpath

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

# use this to upload files
# Modified from https://colab.research.google.com/drive/12QusaaRj_lUwCGDvQNfICpa7kA7_a2dE#scrollTo=G9Fv0wjCMPYY
def upload():
    from google.colab import files
    uploaded = files.upload() 
    for name, data in uploaded.items():
        with open(name, 'wb') as f:
            f.write(data)
            print ('Uploaded file', name)

# use this to download a file  
def download(path):
    from google.colab import files
    files.download(path)

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

# Combine individual prediction files for each image to all_predictions.txt
def combine_predictions(imgs_dir, outfpath):
    # Delete inference images file list
    os.remove(outfpath)
    # Combine inference text files for each image and save to all_predictions.txt
    fns = os.listdir(imgs_dir)
    with open('data/results/all_predictions.txt', 'w') as outfile:
        header = "class_id x y w h img_id"
        outfile.write(header + "\n")
        for fn in fns:
            if '.txt' in fn:
                with open('data/imgs/'+fn) as infile:
                    lines = infile.readlines()
                    newlines = [''.join([x.strip(), ' ' + os.path.splitext(fn)[0] + '\n']) for x in lines]
                    outfile.writelines(newlines)
    # Load all_predictions.txt
    df = pd.read_csv('data/results/all_predictions.txt')
    print("Model predictions by class id: \n", df.head())

    return df

# For uploading an image from url
# Modified from https://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv/
def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_h, im_w = image.shape[:2]

    return image

# Combine tagging files for batches A-D
def combine_tag_files(tags_fpath):
    # Combine tag files for batches A-D
    fpath =  os.path.splitext(tags_fpath)[0]
    base = fpath.rsplit('_',1)[0] + '_'
    exts = ['a.tsv', 'b.tsv', 'c.tsv', 'd.tsv']
    all_filenames = [base + e for e in exts]
    df = pd.concat([pd.read_csv(f, sep='\t', header=0, na_filter = False) for f in all_filenames], ignore_index=True)
    # Choose desired columns for tagging
    df = df[['url', 'img_id', 'class_id']]
    df.rename(columns={'url': 'eolMediaURL', 'img_id': 'identifier', 'class_id': 'tag'}, inplace=True)
    print("\nNew concatenated dataframe with all 4 batches: \n", df[['eolMediaURL', 'tag']].head())

    return df

# Add human-readable text names to numeric classes
def add_class_names(all_predictions):
    # Model predictions with number-coded classes
    numbered_tags = pd.read_csv(all_predictions, header=0, sep=" ")
    numbered_tags.class_id = numbered_tags.class_id - 1 # python counts from 0, Yolo from 1
    #print("\nModel predictions by class id (open images number (YOLO - 1)): \n", numbered_tags)

    # Add class names to model predictions
    classes = pd.read_table('data/openimages.names')
    classes.columns = ['name']
    classes_dict = pd.Series(classes.name.values, index=classes.index).to_dict()
    tags = numbered_tags.copy()
    tags.replace({"class_id":classes_dict}, inplace=True)
    tags['class_id'] = tags['class_id'].astype(str)
    print("\nModel predictions by class id (name): \n", tags)

    return tags

# Add EOL media URL's to named image tags
def add_eolMediaURLs(tags, bundle):
    # Read in EOL 20k image url bundle
    bundle = read_datafile(bundle)
    bundle.columns = ['url']

    # Map eolMediaURLs to tags using image filenames
    img_fns = bundle['url'].apply(lambda x: os.path.splitext((os.path.basename(x)))[0])
    bundle['img_id'] = img_fns
    # Make datatypes consistent for bundle and tags
    bundle['img_id'] = bundle['img_id'].astype("string")
    tags['img_id'] = tags['img_id'].astype("string")
    # Add URLs to tags with img_id as a key
    final_tags = tags.merge(bundle, on='img_id')
    final_tags.reset_index(drop=True, inplace=True)
    final_tags.drop_duplicates(inplace=True, ignore_index=True)
    print("\nModel predictions with EOL media URL's added: \n", final_tags.head())

    return final_tags

# Set filename for saving classification results
def set_outpath(tags_file, cwd):
    tags_file = os.path.splitext(tags_file)[0]
    outpath = cwd + '/data/results/' + tags_file + '.tsv'
    print("Saving results to: \n", outpath)

    return outpath