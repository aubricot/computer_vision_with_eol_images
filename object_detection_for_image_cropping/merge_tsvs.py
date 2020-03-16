# Merge tsv files resulting from object detection of batch images
# 11 Feb 20

import pandas as pd

# File names to be combined
all_filenames = ["chiroptera_det_crops_20000_a.tsv", "chiroptera_det_crops_20000_b.tsv", "chiroptera_det_crops_20000_c.tsv", "chiroptera_det_crops_20000_d.tsv"]

# Combine all files in the list
combined_csv = pd.concat([pd.read_csv(f, sep='\t', header=0) for f in all_filenames])

# Export to csv
combined_csv.to_csv( "chiroptera_det_crops_20000.tsv", index=False, sep='\t')
