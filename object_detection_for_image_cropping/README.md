# Object Detection for Image Cropping 
Testing different existing object detection frameworks using Faster-RCNN (built on Resnet 50 and Inception v2) and SSD (built on MobileNet v2) via the Tensorflow Object Detection API as a method to do customized, large-scale image cropping for different groups of animals for [EOL.org](https://eol.org/pages/2913056/media). Now runs in TF2 and Python 3.   
*Last updated 23 February 2023*  

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/images/banner.jpg" align="middle" width="900" ></a></p>   

<p align="center">
<sub><sup>Images a-c are hosted by Encyclopedia of Life (a. <a href="http://upload.wikimedia.org/wikipedia/commons/a/af/Choeronycteris_mexicana%2C_Mexican_long-tongued_bat_%287371567444%29.jpg"><i>Choeronycteris mexicana</i></a>, licensed under <a href="https://creativecommons.org/licenses/by/2.0/legalcode"></a>CC BY 2.0</a>, b. <a href="https://calphotos.berkeley.edu/cgi/img_query?seq_num=81811&one=T"><i>Hippotion celerio</i></a>, licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/3.0/">CC BY-NC-SA 3.0</a>, c. <a href="https://content.eol.org/data/media/7e/b3/54/542.16276541578.jpg"><i>Cuculus solitarius</i></a> (left) and <a <i>Cossypha caffra</i></a> (right)</a>, licensed under <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC BY-SA 2.0</a>).</sup></sub> 

The tested frameworks differ in their speeds and accuracy: YOLO has been found to be the fastest but least accurate, while Faster-RCNN Resnet 50 was found to be the slowest but most accurate, with MobileNet SSD, R-FCN and Faster-RCNN Inception v2 falling somewhere in between (Lin et al. 2017, Hui 2018, Redmon and Farhadi 2018, [Tensorflow Object Detection Model Zoo]( https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Using the location and dimensions of the detected animals within each framework, images are cropped to square dimensions that are centered around the subject. 

<p align="center">Demo video: Run your own images through the pre-trained EOL object detector in under 2 minutes.</p> 
<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/images/objdet_demo_sm.gif" align="center" width="800" ></a></p> 

## Project Structure
**Aves**
* Pre-trained models are used for "out of the box" inference on images of birds (Aves) of varying dimensions and resolutions. The model with the best trade-off between speed and accuracy - SSD and/or Faster-RCNN - was selected and used to generate cropping data for sets of 1,000 and 20,000 EOL bird images. Detection are consolidated to one box per image, converted to square, and padded by 11% to ensure that beaks are not cropped out.

:arrow_right: :seedling: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/aves/aves_generate_crops_tf2.ipynb) Click here to start generating crops.

**Chiroptera**
* Models are custom trained to detect bats (Chiroptera) from images. For training data, EOL user-generated cropping datasets are used instead of traditional image annotation. Train images are augmented using the [imgaug library](https://github.com/aleju/imgaug) to increase sample size and diversity, reducing overfitting. Model accuracy is compared  for test images using mAP (mean average precision, a standard performance measure used to evaluate object detection models) and AR (average recall). The model with the best trade-off between speed and accuracy - Faster-RCNN - was selected and used to generate cropping data for sets of 1,000 and 20,000 EOL bat images. Detection boxes are consolidated to one box per image and converted to square.

:arrow_right: :seedling: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/chiroptera/chiroptera_generate_crops_tf2.ipynb) Click here to start generating crops.

**Lepidoptera**
* Models are custom trained to detect butterflies and moths (Lepidoptera). For training data, EOL user-generated cropping data is used instead of traditional image annotation.  Train images are augmented using the [imgaug library](https://github.com/aleju/imgaug) to increase sample size and diversity, reducing overfitting. Model accuracy is compared for test images using mAP (mean average precision) and AR (average recall).The model with the best trade-off between speed and accuracy - Faster-RCNN - was selected and used to generate cropping data for a set of 20,000 EOL moth and butterfly images. Detection are consolidated to one box per image and converted to square.   

:arrow_right: :seedling: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/lepidoptera/lepidoptera_generate_crops_tf2.ipynb) Click here to start generating crops.

**Multi-taxa**
* Models are custom trained to detect beetles (Coleoptera), frogs (Anura), carnivores (Carnivora) and snakes & lizards (Squamata). For training data, EOL user-generated cropping data is used instead of traditional image annotation.  Train images are augmented using the [imgaug library](https://github.com/aleju/imgaug) to increase sample size and diversity, reducing overfitting. Model accuracy is compared for test images using mAP (mean average precision) and AR (average recall). The model with the best trade-off between speed and accuracy - Faster-RCNN Inception v2 - was selected and used to generate cropping data for a set of 80,000 EOL images of beetles, frogs, carnivores, snakes & lizards. Detection boxes are consolidated to one box per image and converted to square.

:arrow_right: :seedling: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/multitaxa/multitaxa_generate_crops_tf2.ipynb) Click here to start generating crops.

## Getting Started  
For out of the box inference using EOL pre-trained models, refer to Step 3 of cropping pipelines below.
All files in this repository are run in [Google Colab](https://research.google.com/colaboratory/faq.html). This repository is set up so that each notebook can be run as a standalone script. It is not necessary to clone the entire repository. Instead, you can find project sections that are interesting and directly try for yourself. All needed files and directories are set up within the notebook.

For additional details on steps below, see the [project wiki](https://github.com/aubricot/computer_vision_with_eol_images/wiki).

## Aves
[Pre-trained models are used, so no pre-processing (Step 1) or training (Step 2) steps are needed]  
**Step 3a) Generate square cropping coordinates around birds in EOL images:** For SSD or Faster-RCNN with Tensorflow Object Detection API v2, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/aves/aves_generate_crops_tf2.ipynb). For YOLO in Darkflow (legacy, no longer updated), click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/aves/aves_yolo.ipynb). 

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/images/eagle_crop.jpg" align="center" width="500" ></a></p>   

<p align="center"> 
<sub><sup>Sample output from image (a) to object detection results (b) displayed in a Jupyter Notebook running in Google Colab. Image is hosted by Encyclopedia of Life (<a href="https://content.eol.org/data/media/7e/e7/0f/542.2324933039.jpg"><i>Haliastur indus indus</i></a>, licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/3.0/">CC BY-NC-SA 3.0</a>).</sup></sub>

**Step 3b) Convert bounding boxes to square, centered image cropping coordinates and pad by 2.5%:** 

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/images/bird_cropping.jpg" align="center" width="500" ></a></p>   

<p align="center"> 
<sub><sup>Sample output from original detection bounding box coordinates (a), to square, padded coordinates (b), and the final cropped image thumbnail (c). Image is hosted by Encyclopedia of Life (<a href="https://content.eol.org/data/media/7e/e8/24/542.2339379052.jpg"><i>Asio flammeus sandwichensis</i></a>, licensed under <a href="https://creativecommons.org/licenses/by-nc/2.0/legalcode">CC BY-NC 2.0</a>).</sup></sub>

**Step 3c) Display converted cropping coordinates on images:** 

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/images/6799399_5pad.jpg" align="middle" width="400" ></a></p>   

<p align="center"> 
<sub><sup>Image is hosted by Encyclopedia of Life (<a href="https://content.eol.org/data/media/7e/84/94/542.14577243646.jpg"><i>Charadrius falklandicus</i></a>, licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/2.0/legalcode">CC BY-NC-SA 2.0</a>).</sup></sub>

## Chiroptera
**Step 1) Pre-process train and test datasets:** Test and train datasets (consisting of images and cropping dimensions) are pre-processed and transformed to formatting standards for use with YOLO via Darkflow and SSD and Faster-RCNN object detection models implemented in Tensorflow. All train and test images are also downloaded to Google Drive for future use training and testing. Before reformatting to object detection model standards, training data is augmented using the [imgaug library](https://github.com/aleju/imgaug). Image augmentation is used to increase training data sample size and diversity to reduce overfitting when training object detection models. Both images and cropping coordinates are augmented. The final train dataset consists of augmented and unaugmented images and cropping coordinates. The final test dataset contains only un-augmented images and cropping coordinates. To pre-process Chiroptera train and test datasets, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/chiroptera/chiroptera_preprocessing.ipynb).

**Step 2) Train object detectors:** For SSD or Faster-RCNN with Tensorflow Object Detection API v2, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/chiroptera/chiroptera_train_tf_ssd_rcnn.ipynb). For YOLO in Darkflow (legacy, no longer updated), click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/chiroptera/chiroptera_train_yolo.ipynb). Object detection models are trained until loss<1, for up to 30 hours. YOLO was trained for 3,000 epochs (250,000 steps), SSD for 450,000 steps and Faster-RCNN for 200,000 steps. Object detection models are tested on test images and model performance is measured using mAP (mean average precision) and AR (average recall). SSD and Faster-RCNN models automatically calculate these values. To calculate mAP and AR for YOLO using [mAP](https://github.com/Cartucho/mAP#create-the-ground-truth-files) (legacy, no longer updated), click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/archive/calculate_error_mAP.ipynb).

**Step 3) Generate square cropping coordinates around bats in EOL images:** The model with the best trade-off between speed and accuracy - Faster-RCNN - was selected and used to generate cropping data for sets of 1,000 and 20,000 EOL bat images in the Faster-RCNN training notebook here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/chiroptera/chiroptera_generate_crops_tf2.ipynb).

## Lepidoptera
**Step 1) Pre-process train and test datasets:** Test and train datasets (consisting of images and cropping dimensions) are pre-processed and transformed to formatting standards for use with YOLO via Darkflow and SSD and Faster-RCNN object detection models implemented in Tensorflow. All train and test images are also downloaded to Google Drive for future use training and testing. Before reformatting to object detection model standards, training data is augmented using the [imgaug library](https://github.com/aleju/imgaug). Image augmentation is used to increase training data sample size and diversity to reduce overfitting when training object detection models. Both images and cropping coordinates are augmented. The final train dataset consists of augmented and unaugmented images and cropping coordinates. The final test dataset contains only un-augmented images and cropping coordinates. To pre-process Lepidoptera train and test datasets, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/lepidoptera/lepidoptera_preprocessing.ipynb).

**Step 2) Train object detectors:** For SSD or Faster-RCNN with Tensorflow Object Detection API v2, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/lepidoptera/lepidoptera_train_tf2_ssd_rcnn.ipynb). For YOLO in Darkflow (legacy, no longer updated), click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/lepidoptera/lepidoptera_train_yolo.ipynb). Object detection models are trained until loss<1, for up to 30 hours. YOLO was trained for 3,800 epochs (317,000 steps), SSD for 200,000 steps and Faster-RCNN for 200,000 steps. Object detection models are tested on test images and model performance is measured using mAP (mean average precision) and AR (average recall). SSD and Faster-RCNN models automatically calculate these values. To calculate mAP and AR for YOLO using [mAP](https://github.com/Cartucho/mAP#create-the-ground-truth-files) (legacy, no longer updated), click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/archive/calculate_error_mAP.ipynb).

**Step 3) Generate square cropping coordinates around butterflies & moths in EOL images:** The model with the best trade-off between speed and accuracy - Faster-RCNN - was selected and used to generate cropping data for sets of 1,000 and 20,000 EOL butterfly and moth images in the Faster-RCNN training notebook here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/lepidoptera/lepidoptera_generate_crops_tf2.ipynb).

## Multitaxa
**Step 1) Pre-process train and test datasets:** Test and train datasets (consisting of images and cropping dimensions) exported from multitaxa_split_train_test.ipynb are pre-processed and transformed to formatting standards for use with YOLO via Darkflow and Faster-RCNN and R-FCN object detection models implemented in Tensorflow. All train and test images are also downloaded to Google Drive for future use training and testing. Before reformatting to object detection model standards, training data is augmented using the [imgaug library](https://github.com/aleju/imgaug). Image augmentation is used to increase training data sample size and diversity to reduce overfitting when training object detection models. Both images and cropping coordinates are augmented. The final train dataset consists of augmented and unaugmented images and cropping coordinates. The final test dataset contains only un-augmented images and cropping coordinates. To pre-process train and test datasets for all taxa, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/multitaxa/multitaxa_preprocessing.ipynb).

**Step 2) Train object detectors:** For Faster-RCNN using Resnet 50 or Inception v2 network architecture with Tensorflow Object Detection API v2, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/multitaxa/multitaxa_train_tf2_rcnns.ipynb). For YOLO in Darkflow (legacy, no longer updated) click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/multitaxa/multitaxa_train_yolo.ipynb). Object detection models are trained until loss<1, for up to 30 hours. YOLO was trained for 3,000 epochs (250,000 steps), Faster-RCNN Resnet 50 for 200,000 steps and Faster-RCNN Inception v2 for 200,000 steps. Object detection models are tested on test images and model performance is measured using mAP (mean average precision) and AR (average recall). Faster-RCNN and R-FCN models automatically calculate these values. To calculate mAP and AR for YOLO using [mAP](https://github.com/Cartucho/mAP#create-the-ground-truth-files) (legacy, no longer updated), click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/archive/calculate_error_mAP.ipynb).

**Step 3) Generate square cropping coordinates around beetles, frogs, carnivores, and snakes & lizards in EOL images:** The model with the best trade-off between speed and accuracy - Faster-RCNN Inception v2 - was selected and used to generate cropping data for sets 20,000 EOL beetle, frog, carnivore, snake & lizard images in the Faster-RCNN training notebook here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/multitaxa/multitaxa_generate_crops_tf2.ipynb).
                                                                                              
*--Note on framework versions: Tensorflow pipelines were built in TF1 in 2020 and updated to TF2 in 2021. YOLO pipelines were built with YOLO v2 in Darkflow and are no long being updated because darkflow does not support newer versions of YOLO. For object detection with a native implementation of YOLO v3, see [Object Detection for Image Tagging pipelines](https://github.com/aubricot/computer_vision_with_eol_images/tree/master/object_detection_for_image_tagging)--*

## References
* [Cartucho 2019](https://github.com/Cartucho/mAP). mAP (mean average precision). GitHub.  
* [Dai et al. 2016](https://arxiv.org/abs/1605.06409v2). R-FCN: Object Detection via Region-based Fully Convolutional Networks. 
arXiv:1605.06409v2.   
* [Encyclopedia of Life](http://eol.org)   
* [Huang et al. 2017](https://arxiv.org/abs/1611.10012). Speed/accuracy trade-offs for modern convolutional object detectors. CVPR.
* [Hui 2018](medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359). Object detection: speed and accuracy comparison (Faster R-CNN, R-FCN, SSD, FPN, RetinaNet and YOLOv3). Medium. 27 March 2018. 
* [Jung 2019](https://github.com/aleju/imgaug-doc). imgaug-doc. GitHub.   
* [Lin et al. 2015](https://arxiv.org/pdf/1405.0312.pdf). Microsoft COCO: Common Objects in Context. arXiv:1405.0312.
* [Liu et al. 2015](https://arxiv.org/pdf/1512.02325.pdf). SSD: Single Shot MultiBox Detector.
* [Redmon and Farhadi 2018](https://arxiv.org/pdf/1804.02767.pdf). YOLOv3: An Incremental Improvement.
* [Ren et al. 2016](https://arxiv.org/pdf/1506.01497.pdf). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal 
Networks.
* [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). GitHub.
                                                                                              
<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/images/lep_aug.jpg" align="center" width="600" ></a></p>   

<p align="center">  
<sub><sup> Sample original and augmented images used to increase training data sample size and diversity to reduce overfitting when training object detection models. Image hosted by Encyclopedia of Life (<a href="https://calphotos.berkeley.edu/cgi/img_query?seq_num=81811&one=T"><i>Hippotion celerio</i></a>, licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/3.0/"></a>CC BY-NC-SA 3.0</a>.</sup></sub></p>  
