# Split EOL User-generated Crop Coordinates by taxa and reformat to Pascal VOC Annotation Style
# 28 Mar 20

import csv
import numpy as np
import pandas as pd
import os

# read in user-generated image cropping file
df = pd.read_csv('object_detection_for_image_cropping/data_files/input/Crop_Data/image_crops_withEOL_pk.txt', sep='\t', header=0)
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

# subset of only beetles
taxon1 = crops.loc[crops.ancestry.str.contains('Coleoptera', case=False, na=False)]
taxon1.drop(columns =["ancestry"], inplace = True) 
taxon1['name'] = 'Coleoptera'
print(taxon1.head())
print(taxon1.info())

# export as csv
taxon1.head().to_csv('object_detection_for_image_cropping/data_files/output/Crop_Data/coleoptera_crops_sample.tsv', sep='\t', index=False)
taxon1.to_csv('coleoptera_crops.tsv', sep='\t', index=False)

# subset of only lemurs
taxon2 = crops.loc[crops.ancestry.str.contains('Lemurif', case=False, na=False)]
taxon2.drop(columns =["ancestry"], inplace = True) 
taxon2['name'] = 'Lemuriformes'
print(taxon2.head())
print(taxon2.info())

# export as csv
taxon2.to_csv('object_detection_for_image_cropping/data_files/output/Crop_Data/lemuriformes_crops.tsv', sep='\t', index=False)


# subset of only frogs
taxon3 = crops.loc[crops.ancestry.str.contains('Anura', case=False, na=False)]
taxon3.drop(columns =["ancestry"], inplace = True) 
taxon3['name'] = 'Anura'
print(taxon3.head())
print(taxon3.info())

# export as csv
taxon3.to_csv('object_detection_for_image_cropping/data_files/output/Crop_Data/anura_crops.tsv', sep='\t', index=False)


# subset of only snakes and lizards
taxon4 = crops.loc[crops.ancestry.str.contains('Squamata', case=False, na=False)]
taxon4.drop(columns =["ancestry"], inplace = True) 
taxon4['name'] = 'Squamata'
print(taxon4.head())
print(taxon4.info())

# export as csv
taxon4.to_csv('object_detection_for_image_cropping/data_files/output/Crop_Data/squamata_crops.tsv', sep='\t', index=False)
