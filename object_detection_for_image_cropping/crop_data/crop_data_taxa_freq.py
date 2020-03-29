# Summarize EOL user generated cropping data available by taxa and frequency
# 28 Mar 20

import csv
import numpy as np
import pandas as pd

# Read in EOL user-generated cropping data
df = pd.read_csv('object_detection_for_image_cropping/data_files/input/Crop_Data/image_crops_withEOL_pk.txt', sep='\t', header=0)
print(df.head())

# Make new dataframe from ancestry column, see ex entry below
# {"genus":{"name":"Anguilla","taxon_concept_id":"12992"},"family":{"name":"Anguillidae","taxon_concept_id":"8295"},
# "order":{"name":"Anguilliformes","taxon_concept_id":"8280"},"class":{"name":"Actinopterygii","taxon_concept_id":"1905"},
# "phylum":{"name":"Chordata","taxon_concept_id":"694"},"kingdom":{"name":"Animalia","taxon_concept_id":"1"}}
crops = df[['ancestry']].copy()
print(crops.head())

# Clean special characters from ancestry
crops.ancestry.replace(to_replace=['{', '}', '\(', '\)'], value= '', regex=True, inplace=True)
crops.ancestry.replace(to_replace=["'", '"'], value= '', regex=True, inplace=True)
crops.ancestry.replace('{', '', regex=True, inplace=True)
crops.ancestry.replace(to_replace='name:', value='', regex=True, inplace=True)

# Split ancestry into taxonomic groups
split = pd.DataFrame(crops.ancestry.str.split(",", expand=True).stack(), columns=['a'])
print(split.head())
## Class (but not subclass, superclass or infraclass)
cla = split.a[(split.a.str.contains('class', case=False)==True) & (split.a.str.contains('sub', case=False)==False)
    & (split.a.str.contains('super', case=False)==False) & (split.a.str.contains('infra', case=False)==False)]
cla.replace('class:', '', regex=True, inplace=True)
cla = cla.str.split(" ", expand=True)[0]
print(cla)

## Order (but not suborder, infraorder or superorder)
order = split.a[(split.a.str.contains('order', case=False)==True) & (split.a.str.contains('sub', case=False)==False)
    & (split.a.str.contains('infra', case=False)==False) & (split.a.str.contains('super', case=False)==False)]
order.replace('order:', '', regex=True, inplace=True)
order = order.str.split(" ", expand=True)[0]
print(order)

## Family (but not superfamily or subfamily)
fam = split.a[(split.a.str.contains('family', case=False)==True) & (split.a.str.contains('super', case=False)==False)
    & (split.a.str.contains('sub', case=False)==False)]
fam.replace('family:', '', regex=True, inplace=True)
fam = fam.str.split(" ", expand=True)[0]
print(fam)

## Genus
gen = split.a[split.a.str.contains('genus', case=False)==True]
gen.replace('genus:', '', regex=True, inplace=True)
gen = gen.str.split(" ", expand=True)[0]
print(gen)

# Count frequency of taxonomic groups (class, order, family and genus)
## Pooled groups
# Combine taxonomic groups
comb = pd.concat((cla, order, fam, gen), axis=0, ignore_index=True)
print(comb.head())
# Count frequency
pool = pd.DataFrame(comb.value_counts()).reset_index()
# Sort by decreasing frequency
pool.columns = ["taxon", "freq"]
pool.sort_values(by="freq", axis=0, ascending=False, inplace=True)
print(pool[:10])
# Write results to tsv
pool.to_csv("object_detection_for_image_cropping/data_files/output/Crop_Data/crop_data_taxa_freq_pooled.csv", sep = '\t', index=False, header=True)

## Classes only
# Count frequency
group = pd.DataFrame(cla.value_counts()).reset_index()
# Sort by decreasing frequency
group.columns = ["taxon", "freq"]
group.sort_values(by="freq", axis=0, ascending=False, inplace=True)
print(group[:10])
# Write results to tsv
group.to_csv("object_detection_for_image_cropping/data_files/output/Crop_Data/crop_data_taxa_freq_classes.csv", sep = '\t', index=False, header=True)

## Orders only
# Count frequency
group = pd.DataFrame(order.value_counts()).reset_index()
# Sort by decreasing frequency
group.columns = ["taxon", "freq"]
group.sort_values(by="freq", axis=0, ascending=False, inplace=True)
print(group[:10])
# Write results to tsv
group.to_csv("object_detection_for_image_cropping/data_files/output/Crop_Data/crop_data_taxa_freq_orders.csv", sep = '\t', index=False, header=True)

## Families only
# Count frequency
group = pd.DataFrame(fam.value_counts()).reset_index()
# Sort by decreasing frequency
group.columns = ["taxon", "freq"]
group.sort_values(by="freq", axis=0, ascending=False, inplace=True)
print(group[:10])
# Write results to tsv
group.to_csv("object_detection_for_image_cropping/data_files/output/Crop_Data/crop_data_taxa_freq_families.csv", sep = '\t', index=False, header=True)

## Genera only
# Count frequency
group = pd.DataFrame(gen.value_counts()).reset_index()
# Sort by decreasing frequency
group.columns = ["taxon", "freq"]
group.sort_values(by="freq", axis=0, ascending=False, inplace=True)
print(group[:10])
# Write results to tsv
group.to_csv("object_detection_for_image_cropping/data_files/output/Crop_Data/crop_data_taxa_freq_genera.csv", sep = '\t', index=False, header=True)
