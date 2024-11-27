# Utility functions for running inference
# Last updated 11 Jul 2023 by K Wolcott
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Read in data file exported from "Combine output files A-D" block above
def read_datafile(fpath, sep="\t", header=0, disp_head=True):
    hdr = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'
        }
    try:
        df = pd.read_csv(fpath, sep=sep, header=header, storage_options=hdr)
        if disp_head:
          print("Data header: \n", df.head())
    except FileNotFoundError as e:
        raise Exception("File not found: Enter the path to your file in form field and re-run").with_traceback(e.__traceback__)
    
    return df

# To display loaded image with results
def plot_image_results(i, disp_img, tag):
        _, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(disp_img)
        plt.axis('off')
        plt.title("{}) Image type: {} ".format(i+1, tag))