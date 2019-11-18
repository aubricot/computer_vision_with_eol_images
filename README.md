## object_detection_image_cropping
Using tensorflow-gpu 2.0, CUDA Toolkit 10.0, cuDNN 7.6.0  
Last updated 17 November 2019

Testing different existing object detection frameworks (Tensorflow Object Detection API and YOLO implemented via darkflow) as a method to do customized, large-scale image processing. Using the location and dimensions of the detected animals within each framework, images will be cropped to square dimensions that are centered and padded around the detection box. The frameworks were first tested with pre-trained models for "out of the box" inference on images of birds of varying dimensions and resolutions.

Existing frameworks were modified to be compatible with Windows 10 working on a GPU. Further adjustments were made to extract and modify bounding box coordinates of detected animals to translate these into image cropping coordinates.

For details on installation and getting everything up and running, see the tutorials below that were used and modified to create the current object detection and image cropping pipeline. 

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

# At the end of the above installations and downloads, your directory should look like this: 
TensorFlow
   └─ models
        ├── official
        ├── research
               └── object_detection

        ├── samples
        └── tutorials

# Next, you need to download a model from the Tensorflow Model Zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md "Tensorflow Model Zoo") and unzip the files. 
We used faster_rcnn_resnet50_coco_2018_01_28.tar.gz and ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz. 

# Finally, download the 'test_images' folder from the current repository or create your own 'test_images' folder with your own images (saved as .jpg). 

# Move the folders containing the unzipped model files and test images to 'object_detection.' 
The final directory should look like the one below. 
TensorFlow
   └─ models
        ├── official
        ├── research
               └── object_detection
                         ├── model_downloaded_from_tensorflow_model_zoo
                                           ├── saved_model
                                                    ├── variables/
                                                    └── saved_model.pb
                                           ├── checkpoint
                                           ├── frozen_inference_graph.pb
                                           ├── model.ckpt.data-00000-of-00001
                                           ├── model.ckpt.index
                                           ├── model.ckpt.meta
                                           └── pipeline
                          ├── test_images   
                                  ├── 542.4801468374.jpg
                                  ├── 542.7816025222.jpg
                                  ├── 542.10578857864.jpg
                                  ├── 542.15445377044.jpg
                                  ├── image_info

       ├── samples
       └── tutorials
