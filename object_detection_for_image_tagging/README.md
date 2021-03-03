# Object Detection for Image Tagging
Testing different object detection frameworks as a method to automatically generate tags for plant-pollinator co-ocurrence, insect life stages, and scat/footprints present within EOL images. Such tags will improve user experience by enabling search of information not included in metadata and further facilitate use of EOL’s less structured image content.   
*Last updated 2 March 2021*

## Project Structure
* **Plant/pollintor**: YOLO v3 pre-trained on [Google OpenImages](https://storage.googleapis.com/openimages/web/index.html) using a config file downloaded from [here](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-openimages.cfg) and weights from [here](https://pjreddie.com/media/files/yolov3-openimages.weights). EOL Angiosperm images are run through the model and predictions for 'Butterfly', 'Insect', 'Beetle', 'Ant', 'Bat (Animal)', 'Bird', 'Bee', or 'Invertebrate' were kept and then converted to "pollinator present" during post-processing.
* **Insect life stages**: YOLO v3 pre-trained on [Google OpenImages](https://storage.googleapis.com/openimages/web/index.html) using a config file downloaded from [here](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-openimages.cfg) and weights from [here](https://pjreddie.com/media/files/yolov3-openimages.weights). EOL Insect images are run through the model and predictions for 'Ant', 'Bee', 'Beetle', 'Butterfly', 'Dragonfly', 'Insect', 'Invertebrate', 'Moths and butterflies' were kept and then converted to "adult" during post-processing. Predictions for 'Caterpillar', 'Centipede', 'Worm' were converted to "juvenile" during post-processing.
* **Scat/footprint**: Pre-trained [MobileNet SSD v2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4) and [YOLO v4](https://github.com/AlexeyAB/darknet) models downloaded from [Tensorflow Hub](https://www.tensorflow.org/hub) and [AlexeyAB](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) were re-trained using transfer learning based on advice from the [Tensorflow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification) and [Darknet Github](https://github.com/AlexeyAB/darknet) using iNaturalist images to detect scat or footprints from EOL images. Models never learned despite adjusting augmentation and model hyperparameters for many training sessions. Pipelines should be revisted in the future with different approaches.

Results from object detection models used for the different tasks can be used to inform future large-scale image processing and user features for EOLv3 images.

## Getting Started  
All files in this portion of the repository are run in [Google Colab](https://research.google.com/colaboratory/faq.html) (Google Colaboratory, "a free cloud service, based on Jupyter Notebooks for machine-learning education and research"). Using Colab, everything is run entirely in the cloud (and can link to Google Drive) and no local software or library installs are requried.   

For additional details on steps below, see the [project wiki](https://github.com/aubricot/computer_vision_with_eol_images/wiki).

## Plant/pollinator
**Step 1) Run images through model, post-process and display results:** To run EOL Angiosperm images through pre-trained YOLOv3 model, post-process results (only keep predictions for 'Butterfly', 'Insect', 'Beetle', 'Ant', 'Bat (Animal)', 'Bird', 'Bee', or 'Invertebrate' and convert to "pollinator present"), and display detected tags on images, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/plant_pollinator/plant_poll_cooccurrence_yolov3.ipynb).

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/images/plantpoll_tag_ex.jpg" align="middle" width="500" ></a></p>   

<p align="center">
<sub><sup>Image is hosted by Flickr (<a href="https://www.flickr.com/photos/37089490@N06/3714515042">another flower - insect photo! by thart2009</a>, licensed under <a href="https://creativecommons.org/licenses/by/2.0/?ref=ccsearch&atype=rich">CC BY 2.0</a>).</sup></sub>

## Insect life stages
**Step 1) Run images through model, post-process and display results:** To run EOL Angiosperm images through pre-trained YOLOv3 model, post-process results ( keep predictions for predictions for 'Ant', 'Bee', 'Beetle', 'Butterfly', 'Dragonfly', 'Insect', 'Invertebrate', 'Moths and butterflies' and convert to "adult"; keep pedictions for 'Caterpillar', 'Centipede', 'Worm' and convert to "juvenile"), and display detected tags on images, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/life_stages/insect_lifestages_yolov3.ipynb).

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/images/lifestage_tag_ex.jpg" align="middle" width="500" ></a></p>   

<p align="center">
<sub><sup>Image is hosted by Flickr (<a href="https://c3.staticflickr.com/7/6117/6376974667_631f1dea71_o.jpg">Frangipani Hawkmoth Caterpillar. Pseudosphinx tetrio by galehampshire</a>, licensed under <a href="https://creativecommons.org/licenses/by/2.0/?ref=ccsearch&atype=rich">CC BY 2.0</a>).</sup></sub>

## Scat/footprint
**Step 1) Pre-process training data in Google Drive:** Click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/scat_footprint/scat_footprint_preprocessing.ipynb) to download images from iNaturalist bundles to Google Drive. Then, inspect the number of images per folder and training class. The number of training images per class should be even, so the folder with the fewest images (scat) is used as is. Folders are archived and zipped to download to local machine and add annotations using [labelImg](https://github.com/tzutalin/labelImg). Upload images and annotations to Google Drive from local machine before proceeding with scat_footprint_preprocessing.ipynb. Next, split datasets into train (80% of images and annotations) and test (20% of images and annotations). Then, augment images to increase dataset size and diversity. Finally, scan through files to make sure none are corrupt before moving them to the appropriate folders for training/testing with each model and proceed to Step 2.

**Step 2) Build & train classification models:** After pre-processing the training dataset in Step 1, build and train classification models in YOLOv2 implemented in darkflow by clicking here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/cobject_detection_for_image_tagging/scat_footprint/fscat_footprint_train_yolo_darkflow.ipynb) or in AlexeyAB's fork of YOLOv4 here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/cobject_detection_for_image_tagging/scat_footprint/fscat_footprint_train_yolov4.ipynb). Form fields and dropdown menus within the notebook walk you through model selection, adjusting hyperparameters, training, and the display of trained models. 

Scat/footprint object detection models never learned despite adjusting augmentation and model hyperparameters for many training sessions. If successful approaches are found at a later date, steps for adding tags to images will be included.
 
## References
* [AlexeyAB 2020](https://github.com/AlexeyAB/darknet). darknet. GitHub.
* [Bochkovskiy et al. 2020](https://arxiv.org/abs/2004.10934). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv.
* [Custom training: walkthrough 2020](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough). Tensorflow Core. 12 Jun 2020.
* [Encyclopedia of Life](eol.org)
* [Krasin et al. 2017](https://github.com/openimages). Open-images: A public dataset for large-scale multi-label and multi-class image classification. GitHub.
* [Sharma 2019](medium.com/analytics-vidhya/image-classification-vs-object-detection-vs-image-segmentation-f36db85fe81). Image Classification vs. Object Detection vs. Image Segmentation. Medium. 23 Feb 2020.
* [Tensorflow Image Classification Tutorial 2020](https://www.tensorflow.org/tutorials/images/classification). Tensorflow Core. 10 Jul 2020.
* [Tuzalin 2020](https://github.com/tzutalin/labelImg). labelImg. GitHub.
