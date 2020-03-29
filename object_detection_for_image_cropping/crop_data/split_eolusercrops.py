# Split EOL User-generated Crop Coordinates by taxa and reformat to Pascal VOC Annotation Style
# 29 Mar 20

import csv
import numpy as np
import pandas as pd
import os

# Read in user-generated image cropping file
df = pd.read_csv('object_detection_for_image_cropping/data_files/input/Crop_Data/image_crops_withEOL_pk.txt', sep='\t', header=0)
print(df.head())

# See column names
for col in df.columns: 
    print(col) 

# Make new dataframe, crops, with identifiers, urls, crop_dimensions from df
crops = df[['data_object_id', 'obj_url', 'crop_dimensions','ancestry']].copy()
print(crops.head())

# Split crop_dimensions into separate columns
## Remove/replace characters in crop_dimensions string
crops.crop_dimensions.replace('"', '', regex=True, inplace=True)
crops.crop_dimensions.replace('{', '', regex=True, inplace=True)
crops.crop_dimensions.replace('}', '', regex=True, inplace=True)
crops.crop_dimensions.replace(':', ',', regex=True, inplace=True)
## Split crop_dimensions into new columns for each image and bounding box value
new = crops.crop_dimensions.str.split(",", expand=True)
crops["height"] = new[1]
crops["width"] = new[3]
crops["xmin"] = new[5]
crops["ymin"] = new[7]
crops["xmax"] = new[5].astype(float) + new[9].astype(float) # add cropwidth to xmin, note crops are square so width=height
crops["ymax"] = new[7].astype(float) + new[9].astype(float) # add cropheight to ymin, note crops are square so width=height
## Remove crop_dimensions column
crops.drop(columns =["crop_dimensions"], inplace = True) 
print(crops.head())

# Subset of only beetles (Coleoptera)
taxon1 = crops.loc[crops.ancestry.str.contains('Coleoptera', case=False, na=False)]
taxon1.drop(columns =["ancestry"], inplace = True) 
taxon1['name'] = 'Coleoptera'
print(taxon1.head())
print(taxon1.info())

# Export as csv
taxon1.to_csv('object_detection_for_image_cropping/data_files/output/Crop_Data/coleoptera_crops.tsv', sep='\t', index=False)

# Subset of only lemurs (Lemuriformes)
taxon2 = crops.loc[crops.ancestry.str.contains('Lemurif', case=False, na=False)]
taxon2.drop(columns =["ancestry"], inplace = True) 
taxon2['name'] = 'Lemuriformes'
print(taxon2.head())
print(taxon2.info())

# Export as csv
taxon2.to_csv('object_detection_for_image_cropping/data_files/output/Crop_Data/lemuriformes_crops.tsv', sep='\t', index=False)


# Subset of only frogs (Anura)
taxon3 = crops.loc[crops.ancestry.str.contains('Anura', case=False, na=False)]
taxon3.drop(columns =["ancestry"], inplace = True) 
taxon3['name'] = 'Anura'
print(taxon3.head())
print(taxon3.info())

# Export as csv
taxon3.to_csv('object_detection_for_image_cropping/data_files/output/Crop_Data/anura_crops.tsv', sep='\t', index=False)


# Subset of only snakes and lizards (squamata)
taxon4 = crops.loc[crops.ancestry.str.contains('Squamata', case=False, na=False)]
taxon4.drop(columns =["ancestry"], inplace = True) 
taxon4['name'] = 'Squamata'
print(taxon4.head())
print(taxon4.info())

# Export as csv
taxon4.to_csv('object_detection_for_image_cropping/data_files/output/Crop_Data/squamata_crops.tsv', sep='\t', index=False)
