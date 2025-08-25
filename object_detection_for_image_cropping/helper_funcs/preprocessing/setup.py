# Set up directory structure - Classif for image tagging
# Last updated 7 Aug 2025 by K Wolcott
import os

# Set up directory structure
def setup_dirs(cwd, data_wd, train_wd):
        # Make folders if they don't already exist
        if not os.path.exists(cwd):
                os.makedirs(cwd)
        if not os.path.exists(data_wd):
                os.makedirs(data_wd)
        if not os.path.exists(train_wd):
                os.makedirs(train_wd)
