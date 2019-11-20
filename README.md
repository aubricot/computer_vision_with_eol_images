# Object Detection for Image Cropping
Using tensorflow-gpu 2.0, CUDA Toolkit 10.0, cuDNN 7.6.0  
Last updated 17 November 2019

Testing different existing object detection frameworks (Using Faster-RCNN and SSD detection via the Tensorflow Object Detection API and YOLO via darkflow) as a method to do customized, large-scale image processing. The three frameworks differ in their speeds and accuracy: YOLO has been found to be the fastest but least accurate, while Faster RCNN was found to be the slowest but most accurate, with MobileNet SSD falling somewhere in between (Lin et al. 2017, Hui 2018, Redmon and Farhadi 2018). Using the location and dimensions of the detected animals within each framework, images will be cropped to square dimensions that are centered and padded around the detection box. The frameworks were first tested with pre-trained models for "out of the box" inference on images of birds of varying dimensions and resolutions.

Existing frameworks were modified to be compatible with Windows 10 working on a GPU. Further adjustments were made to extract and modify bounding box coordinates of detected animals to translate these into image cropping coordinates.

For details on installation and getting everything up and running, see the tutorials below that were used and modified to create the current object detection and image cropping pipeline. 

---
---   

## Faster-RCNN and SSD implemented using Tensorflow Object Detection API
First read the official installation instructions from Tensorflow: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-models-installation

Then follow these specific instructions for installing on Windows: https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781  
Modifications: CUDA Toolkit 10.0, cuDNN 7.6.0, tensorflow-gpu 2.0, Protobuf 3.4.0 and Python 3.7 were installed instead of the versions listed above.

Confirm installation and file directory structures through the instructions below before using the notebook in this repository object_detection_for_image_cropping_tf_ssd_rcnn.ipynb. 

You can also see the original Tensorflow Object Detection API demo in Jupyter Notebook that the one in this repository is based off of: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb  
Modifications: The sections "Loader," "Loading Label Maps," and the definition of "show_inference" were modified for use here and other code was replicated verbatim or with only minor variable name changes. 

#### Software Requirements:
Git for Windows
Python 3.7
Visual Studio 2015 with Windows SDK
CUDA Toolkit 10.0
cuDNN 7.6.0
Protobuf 3.4.0

#### Hardware requirements:
Windows 10
compatible NVIDIA GPU (GTX 650 or newer)

#### At the end of the above installations and downloads, your directory should look like this: 
    TensorFlow   

        └─ models  
   
            ├── official  
            ├── research  
              
                  └── object_detection  
        
            ├── samples  
            └── tutorials  

---
### Next, you need to download a model from the Tensorflow Model Zoo  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md and unzip the files. We used faster_rcnn_resnet50_coco_2018_01_28.tar.gz and ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz. 

Finally, download 'object_detection_for_image_cropping_tf_ssd_rcnn.ipynb' and 'sample_img/' from the current repository or create your own 'sample_img/' folder with your own images (saved as .jpg). 

Move the folders containing the unzipped model files, sample images, and object_detection_for_image_cropping_tf_ssd_rcnn.ipynb to 'object_detection'. 

#### The final directory should look like the one below. 
    TensorFlow  

        └─ models  
    
            ├── official  
            ├── research 
        
                   └── object_detection  
                 
                             ├── model_downloaded_from_tensorflow_model_zoo  
                         
                                               ├── saved_model  
                                           
                                                        ├── variables  
                                                        └── saved_model.pb 
                                                    
                                               ├── checkpoint  
                                               ├── frozen_inference_graph.pb  
                                               ├── model.ckpt.data-00000-of-00001  
                                               ├── model.ckpt.index  
                                               ├── model.ckpt.meta  
                                               └── pipeline  
                                           
                              ├── sample_img  
                          
                                      ├── 542.4801468374.jpg  
                                      ├── 542.7816025222.jpg  
                                      ├── 542.10578857864.jpg  
                                      ├── 542.15445377044.jpg   
                              
                              └── object_detection_for_image_cropping_tf_ssd_rcnn.ipynb
           ├── samples  
           └── tutorials  

---   
#### You're now ready to run the object_detection_for_image_cropping_tf_ssd_rcnn.ipynb notebook to try out the code yourself.
---  
---   

## YOLO via Darkflow
### YOLO implemented using Darknet in Tensorflow
First read the official installation instructions for Darkflow: https://github.com/thtrieu/darkflow. You can find documentation for Darknet, the underlying model architecture, here https://pjreddie.com/darknet/.

The installation instructions already work for Windows. 
Modifications: CUDA Toolkit 10.0, cuDNN 7.6.0, tensorflow-gpu 1.15rc and Python 3.7 were installed instead of the versions listed above.

After installation, you need to clone the Darkflow and Darknet repositories to your computer in the folder yolo_imgdetect/.

```
#### download and compile darknet (the underlying framework of YOLO)
git clone https://github.com/pjreddie/darknet
cd darknet
python setup.py build_ext --inplace
```
```
#### install darkflow, the tensorflow implementation of darknet
git clone https://github.com/thtrieu/darkflow.git
pip install .
```

Confirm installation and file directory structures through the instructions below before using the notebook in this repository

#### Software Requirements:
Git for Windows
Python 3.7
Visual Studio 2015 with Windows SDK
CUDA Toolkit 10.0
cuDNN 7.6.0

#### Hardware requirements:
Windows 10
compatible NVIDIA GPU (GTX 650 or newer)

#### At the end of the above installations, your directory should look like this: 
    yolo_imgdetect   

        └─ darknet  
        ├── darkflow  
        ├── ckpt  
        ├── cfg         
        └─ build 
              
---
#### Next you need to download pre-trained sample weight files.
The models are already in darkflow/cfg, but the weights associated with these models need to be downloaded from https://pjreddie.com/media/files/yolov3.weights and https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU.  

After downloading, move them to yolo_imgdetect/bin

Finally, download 'object_detection_for_image_cropping_yolo.ipynb' and 'sample_img/' from the current repository or create a 'sample_img/' folder with your own images (saved as .jpg). 

Move object_detection_for_image_cropping_yolo.ipynb and the folder containing the sample images to 'yolo_imgdetect/'. 

#### The final directory should look like the one below. 
    yolo_imgdetect   

        └─ darknet  
   
        ├── darkflow  
        ├── ckpt  
        ├── cfg   
        
              ├── yolo.cfg
              ├── other_downloaded_files.cfg
              └── yolo-tiny.cfg  
              
        ├── build
        ├── object_detection_for_image_cropping_yolo.ipynb
        ├── sample_img   
        
              ├── 542.4801468374.jpg  
              ├── 542.7816025222.jpg  
              ├── 542.10578857864.jpg  
              └── 542.15445377044.jpg      
              
        └── bin   
        
              ├── yolo.weights
              ├── other_downloaded_files.weights
              └── yolo-tiny.weights

---   
#### You're now ready to run the object_detection_for_image_cropping_yolo.ipynb notebook to try out the code yourself.
---  

#### References:
Hui 2018. Object detection: speed and accuracy comparison (Faster R-CNN, R-FCN, SSD, 
FPN, RetinaNet and YOLOv3). Medium. 27 March 2018. 
medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359.   
Lin et al. 2015. Microsoft COCO: Common Objects in Context. arXiv:1405.0312.  
Liu et al. 2015. SSD: Single Shot MultiBox Detector. arXiv:1512.02325.  
Redmon and Farhadi 2018. YOLOv3: An Incremental Improvement. arXiv:1804.02767.   
Ren et al. 2016. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal 
Networks. arXiv:1506.01497.
