## object_detection_image_cropping
Using tensorflow-gpu 2.0, CUDA Toolkit 10.0, cuDNN 7.6.0  
Last updated 17 November 2019

Testing different existing object detection frameworks (YOLO implemented via darkflow and Tensorflow Object Detection API) as a method to do customized, large-scale image processing. Using the location and dimensions of the detected animals within each framework, images will be cropped to square dimensions that are centered and padded around the detection box. The frameworks were first tested with pre-trained models for "out of the box" inference on images of birds of varying dimensions and resolutions.

Existing frameworks were modified to be compatible with Windows 10 working on a GPU. Further adjustments were made to extract and modify bounding box coordinates of detected animals to translate these into image cropping coordinates.

For details on installation and getting everything up and running, see the tutorials below that were used and modified to create the current object detection and image cropping pipeline.

# YOLO via darkflow
installation: https://github.com/thtrieu/darkflow (with further guidance from https://github.com/thtrieu/darkflow/issues/849)

# Tensorflow Object Detection API
First read the official installation instructions from Tensorflow: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-models-installation

Then follow these specific instructions for installing on Windows: https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781  
Modifications: CUDA Toolkit 10.0, cuDNN 7.6.0, tensorflow-gpu 2.0, Protobuf 3.4.0 and Python 3.7 were installed instead of the versions listed above.

After installation, you are ready to use the notebook in this repository: object_detection_for_image_cropping_tutorial.ipynb. 

You can also see the original Tensorflow Object Detection API demo in Jupyter Notebook that the one in this repository is based off of: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb  
Modifications: The sections "Loader," "Loading Label Maps," and the definition of "show_inference" were modified for use here and other code was replicated verbatim or with only minor variable name changes. 

# Software Requirements:
Git for Windows
Python 3.7
Visual Studio 2015 with Windows SDK
CUDA Toolkit 10.0
cuDNN 7.6.0
Protobuf 3.4.0

# Hardware requirements:
Windows 10
compatible NVIDIA GPU (GTX 650 or newer)
