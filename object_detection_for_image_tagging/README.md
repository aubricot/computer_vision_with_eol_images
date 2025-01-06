# Object Detection for Image Tagging
Testing different object detection frameworks as a method to automatically generate tags for plant-pollinator co-ocurrence, insect life stages, scat/footprints, and humans present within EOL images. Runs in YOLO v3 and Python 3.  
*Last updated 6 January 2025*

## Project Structure
**Plant/pollintor**
* YOLO v3 pre-trained on [Google OpenImages](https://storage.googleapis.com/openimages/web/index.html) using a config file downloaded from [here](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-openimages.cfg) and weights from [here](https://pjreddie.com/media/files/yolov3-openimages.weights). EOL Angiosperm images are run through the model and predictions for 'Butterfly', 'Insect', 'Beetle', 'Ant', 'Bat (Animal)', 'Bird', 'Bee', or 'Invertebrate' were kept and then converted to "pollinator present" during post-processing.

:arrow_right: :seedling: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/plant_pollinator/plant_poll_generate_tags_yolov3.ipynb) Click here to start generating crops.

**Insect life stages**
* YOLO v3 pre-trained on [Google OpenImages](https://storage.googleapis.com/openimages/web/index.html) using a config file downloaded from [here](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-openimages.cfg) and weights from [here](https://pjreddie.com/media/files/yolov3-openimages.weights). EOL Insect images are run through the model and predictions for 'Ant', 'Bee', 'Beetle', 'Butterfly', 'Dragonfly', 'Insect', 'Invertebrate', 'Moths and butterflies' were kept and then converted to "adult" during post-processing. Predictions for 'Caterpillar', 'Centipede', 'Worm' were converted to "juvenile" during post-processing.

:arrow_right: :seedling: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/life_stages/insect_life_stages_generate_tags_yolov3.ipynb) Click here to start generating crops.

**Human Present**
* YOLO v3 pre-trained on [Google OpenImages](https://storage.googleapis.com/openimages/web/index.html) using a config file downloaded from [here](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-openimages.cfg) and weights from [here](https://pjreddie.com/media/files/yolov3-openimages.weights). EOL Chiroptera images are run through the model and predictions for 'Person' or any string containing 'Human' ('Body', 'Eye', 'Head', 'Hand', 'Foot', 'Face', 'Arm', 'Leg', 'Ear', 'Eye', 'Face', 'Nose', 'Beard') were kept and then converted to "human present" during post-processing.   

:arrow_right: :seedling: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/human_present/human_present_generate_tags_yolov3.ipynb) Click here to start generating crops.

**Flower/Fruit**
* YOLO v3 pre-trained on [Google OpenImages](https://storage.googleapis.com/openimages/web/index.html) using a config file downloaded from [here](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-openimages.cfg) and weights from [here](https://pjreddie.com/media/files/yolov3-openimages.weights). EOL Angiosperm images are run through the model and predictions for 'Flower', 'Fruit,' or 'Food' were kept and then converted to "Flower", "Fruit", or "Reproductive structure present" during post-processing. 

:arrow_right: :seedling: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/plant_pollinator/plant_poll_generate_tags_yolov3.ipynb) Click here to start generating crops.


***-Archive-***   
***Scat/footprint***
* *Pre-trained [MobileNet SSD v2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4) and [YOLO v4](https://github.com/AlexeyAB/darknet) models downloaded from [Tensorflow Hub](https://www.tensorflow.org/hub) and [AlexeyAB](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) were re-trained using transfer learning based on advice from the [Tensorflow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification) and [Darknet Github](https://github.com/AlexeyAB/darknet) using iNaturalist images to detect scat or footprints from EOL images. Models never learned despite adjusting augmentation and model hyperparameters for many training sessions. Pipelines should be revisted in the future with different approaches.*

## Getting Started  
All files in this repository are run in [Google Colab](https://research.google.com/colaboratory/faq.html). This repository is set up so that each notebook can be run as a standalone script. It is not necessary to clone the entire repository. Instead, you can just find project sections that are interesting and direclty try for yourself. Any needed files and directories are set up within the notebook.

For additional details on steps below, see the [project wiki](https://github.com/aubricot/computer_vision_with_eol_images/wiki).

## Plant/pollinator
**Step 1) Run images through model, post-process and display results:** To run EOL Angiosperm images through pre-trained YOLOv3 model, post-process results (only keep predictions for 'Butterfly', 'Insect', 'Beetle', 'Ant', 'Bat (Animal)', 'Bird', 'Bee', or 'Invertebrate' and convert to "pollinator present"), and display detected tags on images, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/plant_pollinator/plant_poll_generate_tags_yolov3.ipynb).

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/images/plantpoll_tag_ex.jpg" align="middle" width="500" ></a></p>   

<p align="center">
<sub><sup>Image is hosted by Flickr (<a href="https://www.flickr.com/photos/37089490@N06/3714515042">another flower - insect photo! by thart2009</a>, licensed under <a href="https://creativecommons.org/licenses/by/2.0/?ref=ccsearch&atype=rich">CC BY 2.0</a>).</sup></sub>

## Insect life stages
**Step 1) Run images through model, post-process and display results:** To run EOL Angiosperm images through pre-trained YOLOv3 model, post-process results ( keep predictions for predictions for 'Ant', 'Bee', 'Beetle', 'Butterfly', 'Dragonfly', 'Insect', 'Invertebrate', 'Moths and butterflies' and convert to "Adult"; keep pedictions for 'Caterpillar', 'Centipede', 'Worm' and convert to "Juvenile"), and display detected tags on images, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/life_stages/insect_life_stages_generate_tags_yolov3.ipynb).

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/images/lifestage_tag_ex.jpg" align="middle" width="500" ></a></p>   

<p align="center">
<sub><sup>Image is hosted by Flickr (<a href="https://c3.staticflickr.com/7/6117/6376974667_631f1dea71_o.jpg">Frangipani Hawkmoth Caterpillar. Pseudosphinx tetrio by galehampshire</a>, licensed under <a href="https://creativecommons.org/licenses/by/2.0/?ref=ccsearch&atype=rich">CC BY 2.0</a>).</sup></sub>

## Human Present
**Step 1) Run images through model, post-process and display results:** To run EOL Chiroptera images through pre-trained YOLOv3 model, post-process results (only keep predictions for for 'Person' or any string containing 'Human' ['Body', 'Eye', 'Head', 'Hand', 'Foot', 'Face', 'Arm', 'Leg', 'Ear', 'Eye', 'Face', 'Nose', 'Beard'] and convert to "human present"), and display detected tags on images, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/human_present/human_present_generate_tags_yolov3.ipynb).

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/images/human_present_tag_ex.jpg" align="middle" width="500" ></a></p>   

<p align="center">
<sub><sup>Image is hosted by <a href="https://content.eol.org/data/media/33/de/7d/18.https___www_inaturalist_org_photos_2979499.jpg">EOL</a>.</sup></sub>

## Flower/fruit
**Step 1) Run images through model, post-process and display results:** To run EOL Angiosperm images through pre-trained YOLOv3 model, post-process results (only keep predictions for 'Flower', 'Fruit', or 'Food' and convert to "Flower", "Fruit", or "Reproductive structures present"), and display detected tags on images, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/flower_fruit/flower_fruit_generate_tags_yolov3.ipynb).
 
<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/images/flower_classif_tag_ex.jpg" align="middle" width="500" ></a></p>   

<p align="center">
<sub><sup>Image is hosted by <a href="https://content.eol.org/data/media/81/3c/f3/542.8209936861.jpg">EOL</a>.</sup></sub>

## *Archive - Scat/footprint*
***Step 1) Pre-process training data in Google Drive:*** *Click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/scat_footprint/scat_footprint_preprocessing.ipynb) to download images from iNaturalist bundles to Google Drive. Then, inspect the number of images per folder and training class. The number of training images per class should be even, so the folder with the fewest images (scat) is used as is. Folders are archived and zipped to download to local machine and add annotations using [labelImg](https://github.com/tzutalin/labelImg). Upload images and annotations to Google Drive from local machine before proceeding with scat_footprint_preprocessing.ipynb. Next, split datasets into train (80% of images and annotations) and test (20% of images and annotations). Then, augment images to increase dataset size and diversity. Finally, scan through files to make sure none are corrupt before moving them to the appropriate folders for training/testing with each model and proceed to Step 2.*

***Step 2) Build & train classification models:*** *After pre-processing the training dataset in Step 1, build and train classification models in YOLOv2 implemented in darkflow by clicking here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/cobject_detection_for_image_tagging/scat_footprint/scat_footprint_train_yolo_darkflow.ipynb) or in AlexeyAB's fork of YOLOv3 here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/cobject_detection_for_image_tagging/scat_footprint/scat_footprint_train_yolov3.ipynb). Form fields and dropdown menus within the notebook walk you through model selection, adjusting hyperparameters, training, and the display of trained models.* 

*Scat/footprint object detection models never learned despite adjusting augmentation and model hyperparameters for many training sessions. If successful approaches are found at a later date, steps for adding tags to images will be included.*

## Data and model availability
EOL image tags and square cropping coordinates produced using these pipelines are available on [Zenodo](https://zenodo.org/communities/eol/records?q=computer%20vision&l=list&p=1&s=10&sort=bestmatch). EOL trained models are currently set to directly download within Colab Notebooks, but will be deposited on other websites for download shortly.
 
## References
* [AlexeyAB 2020](https://github.com/AlexeyAB/darknet). darknet. GitHub.
* [Custom training: walkthrough 2020](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough). Tensorflow Core. 12 Jun 2020.
* [Encyclopedia of Life](eol.org)
* [Krasin et al. 2017](https://github.com/openimages). Open-images: A public dataset for large-scale multi-label and multi-class image classification. GitHub.
* [Sharma 2019](medium.com/analytics-vidhya/image-classification-vs-object-detection-vs-image-segmentation-f36db85fe81). Image Classification vs. Object Detection vs. Image Segmentation. Medium. 23 Feb 2020.
* [Redmon and Farhadi 2018](https://arxiv.org/abs/1804.02767). YOLOv3: An Incremental Improvement. arXiv.
* [Tensorflow Image Classification Tutorial 2020](https://www.tensorflow.org/tutorials/images/classification). Tensorflow Core. 10 Jul 2020.
* [Tuzalin 2020](https://github.com/tzutalin/labelImg). labelImg. GitHub.
