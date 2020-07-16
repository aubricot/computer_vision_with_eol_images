# Preprocessing PlantCLEF 2016 dataset
# 15 Jul 2020

import os, random, shutil

# Directions before running this python script (Steps 1-3):
# 1) Download PlantCLEF 2016 dataset from https://www.imageclef.org/lifeclef/2016/plant

# 2) Unzip folder containing dataset

# 3) Move all xmls to separate folder - Run command below in terminal
#mv flowers/PlantCLEF2015TrainingData/train/*.xml flowers/PlantCLEF2015TrainingData/xml

# Next run this python script:

# Randomly pick 6,000 images to use for training data and move to new folder
# From https://github.com/THE-PHOENIX-777-TDW/Random-File-Picker
# User prompts
# input = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/train'
source=input("Enter the Source Directory : ")
# input = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/image_subset'
dest=input("Enter the Destination Directory : ")
no_of_files=int(input("Enter The Number of Files To Select : "))

print("%"*25+"{ Details Of Transfer }"+"%"*25)
print("\n\nList of Files Moved to %s :-"%(dest))

# Randomly choose files
for i in range(no_of_files):
    # Variable random_file stores the name of the random file chosen
    random_file=random.choice(os.listdir(source))
    print("%d} %s"%(i+1,random_file))
    source_file="%s\%s"%(source,random_file)
    dest_file=dest
    # Move file from one directory to another
    shutil.move(source_file,dest_file)

print("\n\n"+"$"*33+"[ Files Moved Successfully ]"+"$"*33)

# Next move all XMLs matching randomly selected JPGs to new folder
# Directory where images are saved
path = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/image_subset'
files = os.listdir(path)

# Move corresponding xmls to new folder
for file in files: 
  base = os.path.splitext(file)[0]
  folder1 = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/xml/'
  folder2 = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/xml_subset/'
  fname = str(base) + '.xml'
  fpath = str(folder1) + str(fname) 
  if os.path.isfile(fpath):
      shutil.move(fpath, folder2)
      print(fname)

# Sort xml files into folders based on Image Class
# Class (Fruit, Flower, Stem, Branch, Entire, Leaf) is noted in xmls
import xml.etree.ElementTree as ET

path = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/xml_subset'
xentire = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/xmls/entire'
xfruit = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/xmls/fruit'
xbranch = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/xmls/branch'
xstem = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/xmls/stem'
xleaf = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/xmls/leaf'
xflower = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/xmls/flower'

for fname in os.listdir(path):
    fpath = os.path.normpath(os.path.join(path, fname))
    tree = ET.parse(fpath)
    root = tree.getroot()
    classname = [w.text for w in root.findall('./Content')]
    if classname==['Entire']:
        shutil.move(fpath, xentire)
    elif classname==['Fruit']:
        shutil.move(fpath, xfruit)
    elif classname==['Branch']:
        shutil.move(fpath, xbranch)
    elif classname==['Stem']:
        shutil.move(fpath, xstem)
    elif classname==['Flower']:
        shutil.move(fpath, xflower)
    elif classname==['LeafScan']:
        shutil.move(fpath, xleaf)
    elif classname==['Leaf']:
        shutil.move(fpath, xleaf)

# Sort images into folders based on Image Class
# Directories for images
imagepath = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/image_subset/'
entire = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/images/entire'
fruit = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/images/fruit'
branch = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/images/branch'
stem = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/images/stem'
leaf = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/images/leaf'
flower = 'classification_for_image_tagging/data_files/input/flowers/PlantCLEF2015TrainingData/images/flower'

# Move xmls corresponing to images to class specific folders
# Entire
files = os.listdir(xentire)
for file in files: 
  xname = os.path.split(file)[1]
  base = os.path.splitext(file)[0]
  imname = str(base) + '.jpg'
  impath = str(imagepath) + str(imname) 
  if os.path.isfile(impath):
      shutil.move(impath, entire)

# Fruit
files = os.listdir(xfruit)
for file in files: 
  xname = os.path.split(file)[1]
  base = os.path.splitext(file)[0]
  imname = str(base) + '.jpg'
  impath = str(imagepath) + str(imname) 
  if os.path.isfile(impath):
      shutil.move(impath, fruit)

# Branch
files = os.listdir(xbranch)
for file in files: 
  xname = os.path.split(file)[1]
  base = os.path.splitext(file)[0]
  imname = str(base) + '.jpg'
  impath = str(imagepath) + str(imname) 
  if os.path.isfile(impath):
      shutil.move(impath, branch)

# Stem
files = os.listdir(xstem)
for file in files: 
  xname = os.path.split(file)[1]
  base = os.path.splitext(file)[0]
  imname = str(base) + '.jpg'
  impath = str(imagepath) + str(imname) 
  if os.path.isfile(impath):
      shutil.move(impath, stem)

# Leaf
files = os.listdir(xleaf)
for file in files: 
  xname = os.path.split(file)[1]
  base = os.path.splitext(file)[0]
  imname = str(base) + '.jpg'
  impath = str(imagepath) + str(imname) 
  if os.path.isfile(impath):
      shutil.move(impath, leaf)

# Flower
files = os.listdir(xflower)
for file in files: 
  xname = os.path.split(file)[1]
  base = os.path.splitext(file)[0]
  imname = str(base) + '.jpg'
  impath = str(imagepath) + str(imname) 
  if os.path.isfile(impath):
      shutil.move(impath, flower)