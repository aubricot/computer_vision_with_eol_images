# Object Detection for Image Cropping 
To test object detectors without any installs or downloads, go to the Google Colab links below.   
*Last updated 2 December 2019*

Testing different existing object detection frameworks (Using Faster-RCNN and SSD detection via the Tensorflow Object Detection API and YOLO via Darkflow) as a method to do customized, large-scale image processing. The three frameworks differ in their speeds and accuracy: YOLO has been found to be the fastest but least accurate, while Faster RCNN was found to be the slowest but most accurate, with MobileNet SSD falling somewhere in between (Lin et al. 2017, Hui 2018, Redmon and Farhadi 2018). Using the location and dimensions of the detected animals within each framework, images will be cropped to square dimensions that are centered and padded around the detection box. The frameworks are tested with pre-trained models for "out of the box" inference on images of birds of varying dimensions and resolutions, but will be modified and fine-tuned in future efforts.

## Getting Started  
We recommend first testing and viewing the code within Google Colab (Google Colaboratory, "a free cloud service, based on Jupyter Notebooks for machine-learning education and research"). If using Colab, you run everything entirely in the cloud and no local software or library installs are requried. If running locally and using a GPU, there are several softwares that need to be installed first and take up ~10 GB and a few workarounds are required if running on a Windows OS. Working in the cloud eliminates these problems and makes it easier to collaborate if multiple users are on different operating systems. If you prefer to use your local machine, please refer to the [Tensorflow Object Detection API Tutorial] (https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb). 

**Step 1) Object Detection:** For YOLO in Darkflow, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/object_detection_for_image_cropping/blob/master/COLAB_object_detection_for_image_cropping_yolo.ipynb) or for SSD or Faster-RCNN with Tensorflow Object Detection API, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/object_detection_for_image_cropping/blob/master/COLAB_object_detection_for_image_cropping_tf_ssd_rcnn.ipynb).

You can also view the COLAB notebooks directly by clicking on the file names in the repository (ex: COLAB_object_detection_for_image_cropping_[model_name].ipynb). 

**Step 2) Convert bounding boxes to square, centered image cropping coordinates:** To convert bounding box coordinates to EOL crop coodinate formatting standards, use convert_bboxdims.py on sample_crops.tsv exported from Google Colab in Step 1.

**Step 3) Display converted cropping coordinates on images:** To display converted crop coordinates on images and verify that the transformations in convert_bboxdims.py are appropriate (or to fine tune accordingly), click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/object_detection_for_image_cropping/blob/master/crop_coords_display_test.ipynb).

---
---   

#### References for different object detection models:
Hui 2018. Object detection: speed and accuracy comparison (Faster R-CNN, R-FCN, SSD, 
FPN, RetinaNet and YOLOv3). Medium. 27 March 2018. 
medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359.   
Lin et al. 2015. Microsoft COCO: Common Objects in Context. arXiv:1405.0312.  
Liu et al. 2015. SSD: Single Shot MultiBox Detector. arXiv:1512.02325.  
Redmon and Farhadi 2018. YOLOv3: An Incremental Improvement. arXiv:1804.02767.   
Ren et al. 2016. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal 
Networks. arXiv:1506.01497.
