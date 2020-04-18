# Computer Vision with EOL v3 Images 
Testing different computer vision methods (object detection, image classification) to do customized, large-scale image processing for [Encyclopedia of Life v3 database images](https://eol.org/pages/2913056/media).   
*Last updated 15 April 2020*

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/images/banner.jpg" align="middle" width="900" ></a></p>   

<p align="center">
<sub><sup>Images a-c are hosted by Encyclopedia of Life (a. <a href="http://upload.wikimedia.org/wikipedia/commons/a/af/Choeronycteris_mexicana%2C_Mexican_long-tongued_bat_%287371567444%29.jpg"><i>Choeronycteris mexicana</i></a>, licensed under <a href="https://creativecommons.org/licenses/by/2.0/legalcode"></a>CC BY 2.0</a>, b. <a href="https://calphotos.berkeley.edu/cgi/img_query?seq_num=81811&one=T"><i>Hippotion celerio</i></a>, licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/3.0/">CC BY-NC-SA 3.0</a>, c. <a href="https://content.eol.org/data/media/7e/b3/54/542.16276541578.jpg"><i>Cuculus solitarius</i></a> (left) and <a <i>Cossypha caffra</i></a> (right)</a>, licensed under <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC BY-SA 2.0</a>).</sup></sub>

The Encyclopedia of Life (EOL) is an online biodiversity resource that seeks to provide information about all ~1.9 million species known to science. A goal for the latest version of EOL (v3) is to better leverage the older, less structured image content. To improve discoverability and display of EOL images, automated image processing pipelines that use computer vision with the goal of being scalable to millions of diverse images are being developed and tested.   

## Project Structure
<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/images/3093378.jpg" align="right" width="400" ></a></p> 

* **Object detection for image cropping**: Three object detection frameworks (Faster-RCNN Resnet 50 and either SSD/R-FCN/Faster-RCNN Inception v2 detection via the Tensorflow Object Detection API and YOLO via Darkflow) were used to perform customized, large-scale image processing for different groups of animals (birds, bats, butterflies & moths, beetles, frogs, carnivores, snakes & lizards) found in the Encyclopedia of Life v3 database. The three frameworks differ in their speeds and accuracy: YOLO has been found to be the fastest but least accurate, while Faster RCNN was found to be the slowest but most accurate, with MobileNet SSD and R-FCN falling somewhere in between (Lin et al. 2017, Hui 2018, Redmon and Farhadi 2018). The model with the best trade-off between speed and accuracy for each group was selected to generate final cropping data for EOL images. The location and dimensions of the detected animals within each framework are used to crop images to square dimensions that are centered and padded around the detection box. For birds, pre-trained object detection models were used. For bats and butterflies & moths, object detection models were custom-trained to detect one class (either bats or butterflies & moths) using EOL user-generated cropping data (square coordinates around animal(s) of interest within each photo). For beetles, frogs, carnivores and snakes & lizards, object detection models were custom-trained to detect all classes simultaneously using EOL user-generated cropping data.  

<p align="right"> 
<sub><sup> Screenshot of object detection results using trained multitaxa detector model displayed in a Jupyter Notebook running in Google Colab. Image is hosted by Encyclopedia of Life (<a href="https://content.eol.org/data/media/28/09/c5/18.https___www_inaturalist_org_photos_1003389.jpg"><i>Lampropeltis californiae</i></a>, licensed under <a href="https://creativecommons.org/licenses/by-nc/4.0/"></a>CC BY-NC 4.0</a>.</sup></sub></p>  

* **Classification for image tagging**: [Future steps, not yet in progress] Image tagging will further improve EOL user experience, allowing users to search for features within images that are not already noted in metadata. Image classification can be used to label images with flowers present, images of maps, collection labels, and illustrations, and to generate image quality ratings. EOL user-generated cropping and image quality datasets will be used to custom train image classifiers. 

Results from object detection and image classification tasks will be used to inform future computer vision efforts and large-scale image processing for EOLv3 images. 

## Getting Started  
Except for .py files, all files in this repository are run in [Google Colab](https://research.google.com/colaboratory/faq.html) (Google Colaboratory, "a free cloud service, based on Jupyter Notebooks for machine-learning education and research"). Using Colab, everything runs entirely in the cloud (and can link to Google Drive) and no local software or library installs are requried. If running locally and using a GPU, there are several softwares that need to be installed first and take up ~10 GB and a few workarounds are required if running on a Windows OS. Working in the cloud eliminates these problems and makes it easier to collaborate if multiple users are on different operating systems. To use your local machine for object detection, refer to the [Tensorflow Object Detection API Tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb).

## References
* [Encyclopedia of Life](http://eol.org) 
* [Hui 2018](medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359). Object detection: speed and accuracy comparison (Faster R-CNN, R-FCN, SSD, 
FPN, RetinaNet and YOLOv3). Medium. 27 March 2018. [medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359].   
* [Lin et al. 2015](https://arxiv.org/pdf/1405.0312.pdf). Microsoft COCO: Common Objects in Context.  
* [Redmon and Farhadi 2018](https://arxiv.org/pdf/1804.02767.pdf). YOLOv3: An Incremental Improvement.    
