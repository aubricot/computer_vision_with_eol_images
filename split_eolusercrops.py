# Split EOL User-generated Crop Coordinates by taxa and reformat to Pascal VOC Annotation Style
# 17 feb 20

import csv
import numpy as np
import pandas as pd
import os

# read in user-generated image cropping file
df = pd.read_csv('image_crops_withEOL_pk.txt', sep='\t', header=0)
print(df.head())

# see column names
for col in df.columns: 
    print(col) 

# make new dataframe, crops, with identifiers, urls, crop_dimensions from df
crops = df[['data_object_id', 'obj_url', 'crop_dimensions','ancestry']].copy()
print(crops.head())

# split crop_dimensions into separate columns
## remove/replace characters in crop_dimensions string
crops.crop_dimensions.replace('"', '', regex=True, inplace=True)
crops.crop_dimensions.replace('{', '', regex=True, inplace=True)
crops.crop_dimensions.replace('}', '', regex=True, inplace=True)
crops.crop_dimensions.replace(':', ',', regex=True, inplace=True)
## split crop_dimensions into new columns for each image and bounding box value
new = crops.crop_dimensions.str.split(",", expand=True)
crops["height"] = new[1]
crops["width"] = new[3]
crops["xmin"] = new[5]
crops["ymin"] = new[7]
crops["xmax"] = new[5].astype(float) + new[9].astype(float) # add cropwidth to xmin, note crops are square so width=height
crops["ymax"] = new[7].astype(float) + new[9].astype(float) # add cropheight to ymin, note crops are square so width=height
## remove crop_dimensions column
crops.drop(columns =["crop_dimensions"], inplace = True) 
print(crops.head())

# subset of only bats
chirocrops = crops.loc[crops.ancestry.str.contains('Chiroptera', case=False, na=False)]
chirocrops.drop(columns =["ancestry"], inplace = True) 
chirocrops['name'] = 'Chiroptera'
print(chirocrops.head())
print(chirocrops.info())

# export as csv
chirocrops.head().to_csv('chiroptera_crops_sample.tsv', sep='\t', index=False)
chirocrops.to_csv('chiroptera_crops.tsv', sep='\t', index=False)

# subset of only bats
lepcrops = crops.loc[crops.ancestry.str.contains('Lepidoptera', case=False, na=False)]
lepcrops.drop(columns =["ancestry"], inplace = True) 
lepcrops['name'] = 'Lepidoptera'
print(lepcrops.head())
print(lepcrops.info())

# export as csv
lepcrops.to_csv('lepidoptera_crops.tsv', sep='\t', index=False)
