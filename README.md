# Computer Vision with EOL v3 Images 
Testing different computer vision methods (object detection, image classification) to do customized, large-scale image processing for [Encyclopedia of Life v3 database images](https://eol.org/pages/2913056/media) (square, centered crops; image content tags; etc). Runs in Tensorflow 2 and Python 3.  
*Last updated 14 September 2025*

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/images/banner.jpg" align="middle" width="900" ></a></p>   

<p align="center">
<sub><sup>Images a-c are hosted by Encyclopedia of Life (a. <a href="http://upload.wikimedia.org/wikipedia/commons/a/af/Choeronycteris_mexicana%2C_Mexican_long-tongued_bat_%287371567444%29.jpg"><i>Choeronycteris mexicana</i></a>, licensed under <a href="https://creativecommons.org/licenses/by/2.0/legalcode"></a>CC BY 2.0</a>, b. <a href="https://calphotos.berkeley.edu/cgi/img_query?seq_num=81811&one=T"><i>Hippotion celerio</i></a>, licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/3.0/">CC BY-NC-SA 3.0</a>, c. <a href="https://content.eol.org/data/media/7e/b3/54/542.16276541578.jpg"><i>Cuculus solitarius</i></a> (left) and <a <i>Cossypha caffra</i></a> (right)</a>, licensed under <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC BY-SA 2.0</a>).</sup></sub>

The Encyclopedia of Life (EOL) is an online biodiversity resource that seeks to provide information about all ~1.9 million species known to science. A goal for the latest version of EOL (v3) is to better leverage the older, less structured image content. To improve discoverability and display of EOL images, automated image processing pipelines that use computer vision with the goal of being scalable to millions of diverse images are being developed and tested. 

## Project Structure
<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/images/3093378.jpg" align="right" width="400" ></a></p> 

**Object detection for image cropping**

Three object detection frameworks (Faster-RCNN Resnet 50 and either SSD/R-FCN/Faster-RCNN Inception v2 <a href="#note1" id="note1ref"><sup>1</sup></a> detection via the Tensorflow Object Detection API and YOLO v3 <a href="#note2" id="note2ref"><sup>2</sup></a> via Darkflow) were used to perform square cropping for EOL images of different groups of animals (birds, bats, butterflies & moths, beetles, frogs, carnivores, snakes & lizards) by using transfer learning and/or fine-tuning. 

Frameworks differ in their speeds and accuracy: YOLO is the fastest but least accurate, while Faster RCNN is the slowest but most accurate, with MobileNet SSD and R-FCN falling somewhere in between <a href="#note2" id="note2ref"><sup>2</sup></a> <a href="#note3" id="note3ref"><sup>3</sup></a> <a href="#note4" id="note4ref"><sup>4</sup></a>. The model with the best trade-off between speed and accuracy for each group was selected to generate final cropping data for EOL images. 

After detection, bounding boxes of detected animals are converted to square, centered cropping coordinates in order to standardize heterogenous image gallery displays.   

* For <ins>birds</ins>, pre-trained object detection models were used to detect birds.   
* For <ins>bats and butterflies & moths</ins>, object detection models were custom-trained to detect one class (either bats or butterflies & moths) using EOL user-generated cropping data (square coordinates around animal(s) of interest within each photo).    
* For <ins>beetles, frogs, carnivores and snakes & lizards</ins>, object detection models were custom-trained to detect all classes simultaneously using EOL user-generated cropping data.  


:arrow_right: :seedling: [Click here](https://github.com/aubricot/computer_vision_with_eol_images/tree/master/object_detection_for_image_cropping) to get started.

<p align="center">Demo video: Run your own images through the pre-trained EOL object detector in under 2 minutes.</p> 
<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/images/objdet_demo_sm.gif" align="center" width="800" ></a></p> 

<p align="center"> 
<sub><sup> Object detection results using trained multitaxon detector model displayed in a Google Colab Notebook. Image is hosted by Encyclopedia of Life (<a href="https://content.eol.org/data/media/28/09/c5/18.https___www_inaturalist_org_photos_1003389.jpg"><i>Lampropeltis californiae</i></a>, licensed under <a href="https://creativecommons.org/licenses/by-nc/4.0/"></a>CC BY-NC 4.0</a>.</sup></sub></p>  

# 
**Classification for image tagging** 

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/images/classification_example.jpg" align="right" width="300" ></a></p> 

Two classification frameworks (MobileNetSSD v2 <a href="#note11" id="note11ref"><sup>11</sup></a>, Inception v3 <a href="#note5" id="note5ref"><sup>5</sup></a>) were used to perform image tagging for different classes of EOL images (flowers, maps/labels/illustrations, image ratings) by using transfer learning and/or fine-tuning. 

Frameworks differ in their speed and accuracy: MobileNetSSD v2 is faster, smaller, and less accurate and Inception v3 is slower, larger, and more accurate <a href="#note5" id="note5ref"><sup>5</sup></a> <a href="#note6" id="note6ref"><sup>6</sup></a>. The model with the best trade-off between speed and accuracy for each group was selected to generate final tagging data for EOL images. 

While object detection includes classification and localization of the object of interest, image classification only includes the former step <a href="#note7" id="note7ref"><sup>7</sup></a>. Classification is used to identify images with flowers present, images of maps/collection labels/illustrations, and to generate image quality ratings. These tags will allow users to search for features not already included in image metadata.

* For the <ins>flower classifier</ins>, models were trained to classify images into flower, fruit, entire, branch, stem or leaf using the [PlantCLEF 2016 Image dataset](https://www.imageclef.org/lifeclef/2016/plant) as training data <a href="#note8" id="note8ref"><sup>8</sup></a>. 
* For the <ins>flower/fruit classifier</ins>, models were trained to classify images into flower/fruit or not flower/fruit using manually-sorted EOL images as training data.    
* For the <ins>image type classifier</ins>, models were trained to classify images into map, herbarium sheet, phylogeny, illustration, or none using Wikimedia commons, Flickr BHL, and EOL images as training data.    
* For the <ins>image rating classifier</ins>, models were trained to classify image quality rating classes 1-5 (worst to best) using EOL user generated training data.   

:arrow_right: :seedling: [Click here](https://github.com/aubricot/computer_vision_with_eol_images/tree/master/classification_for_image_tagging) to get started.

<p align="center">
<sub><sup>Image classification results using trained flower/fruit classification model displayed in a Google Colab Notebook. Image is hosted by Encyclopedia of Life (<a href="https://content.eol.org/data/media/66/a1/2a/509.63397702.jpg"><i>Leucopogon tenuicaulis</i></a>, licensed under <a href="https://creativecommons.org/licenses/by/3.0/">CC BY 3.0</a>).</sup></sub>

#
**Object detection for image tagging**    

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/images/plantpoll_tag_ex.jpg" align="right" width="400" ></a></p>   

Three object detection frameworks (YOLO v3 in darknet <a href="#note9" id="note9ref"><sup>9</sup></a>, MobileNetSSD v2 <a href="#note10" id="note10ref"><sup>10</sup></a>, and YOLO v4 <a href="#note10" id="note10ref"><sup>10</sup></a>) were used to perform image tagging for different classes of EOL images (flowers, insects, mammals/amphibians/reptiles/birds). 

Frameworks differ in their speeds and accuracy: YOLO v4 is the fastest with intermediate accuracy, MobileNetSSD v2 is intermediate speed and accuracy, and YOLO v3 is somewhere in between <a href="#note10" id="note10ref"><sup>10</sup></a> <a href="#note3" id="note3ref"><sup></sup></a> <a href="#note6" id="note6ref"><sup>6</sup></a>). The model with the best trade-off between speed and accuracy for each group was selected to generate final tagging data for EOL images.

For tagging, only the class of detected objects are kept and their locations are discarded. Object detection is used to identify plant-insect coocurrence, insect life stage, the presence of mammal, amphibian, reptile, or bird scat and/or footprints, and when a human (or body part, like 'hand') is present. These tags will allow users to search for features not already included in image metadata. 
  
* For <ins>plant-insect coocurrence</ins>, a model pre-trained on Google OpenImages <a href="#note12" id="note12ref"><sup>12</sup></a> was used. EOL images are run through the model and predictions for 'Butterfly', 'Insect', 'Beetle', 'Ant', 'Bat (Animal)', 'Bird', 'Bee', or 'Invertebrate' were kept and then converted to "insect visitor" during post-processing.    
* For <ins>insect life stages</ins>, a model pre-trained on Google OpenImages <a href="#note12" id="note12ref"><sup>12</sup></a> was used. EOL images are run through the model and predictions for 'Ant', 'Bee', 'Beetle', 'Butterfly', 'Dragonfly', 'Insect', 'Invertebrate', 'Moths and butterflies' were kept and then converted to "adult" during post-processing. Predictions for 'Caterpillar', 'Centipede', 'Worm' were converted to "juvenile" during post-processing.
* For <ins>scat/footprint present</ins>, models were custom-trained to detect scat or footprints from EOL images, but never learned despite adjusting augmentation and model hyperparameters for many training sessions. Pipelines and datasets should be revisted in the future with different approaches.
* For <ins>human present</ins>, a model pre-trained on Google OpenImages <a href="#note12" id="note12ref"><sup>12</sup></a> was used. EOL images are run through the model and predictions for 'Person' or any string containing 'Human' ('Body', 'Eye', 'Head', 'Hand', 'Foot', 'Face', 'Arm', 'Leg', 'Ear', 'Eye', 'Face', 'Nose', 'Beard') were kept and then converted to "human present" during post-processing. 

:arrow_right: :seedling: [Click here](https://github.com/aubricot/computer_vision_with_eol_images/tree/master/object_detection_for_image_tagging) to get started.

<p align="center">
<sub><sup>Object detection for image tagging results using pre-trained plant-insect coocurrence model displayed in a Google Colab Notebook. Image is hosted by Flickr (<a href="https://www.flickr.com/photos/37089490@N06/3714515042">another flower - insect photo! by thart2009</a>, licensed under <a href="https://creativecommons.org/licenses/by/2.0/?ref=ccsearch&atype=rich">CC BY 2.0</a>).</sup></sub>

#
**Utils**    
This folder contains Colab Notebooks and Google Chrome developer console scripts with useful functions for building on existing EOL computer vision pipelines or for developing your own from scratch.
#
  
## Getting Started  
All files in this repository are run in [Google Colab](https://research.google.com/colaboratory/faq.html)*. This repository is set up so that each notebook can be run as a standalone script. It is not necessary to clone the entire repository. Instead, you can navigate project sections (ie. GitHub folders) that are interesting and directly try the notebooks for yourself! All needed files and directories are set up within the notebook.

For additional details on steps below, see the [project wiki](https://github.com/aubricot/computer_vision_with_eol_images/wiki).

>New to Google Colab?   
>Google Colaboratory is "a free cloud service, based on Jupyter Notebooks for machine-learning education and research." Notebooks run >entirely on VMs in the cloud and links to you Google Drive for accessing files. This means no local software or library installs are >requried. If running locally and using a GPU, there are several softwares that need to be installed first and take up ~10 GB and a >few workarounds are required if running on a Windows OS. Working in the cloud eliminates these problems and makes it easier to >collaborate if multiple users are on different operating systems. If you prefer to use your local machine for object detection, refer to the [Tensorflow Object Detection API Tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb).*   

## Data and model availability
| Resource Type                      | Description                                                                                      | Link                                                                                                       |
|-----------------------------------|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
|  **Classification for Image Tagging** |                                                                                                  |                                                                                                            |
| 🏷️ Image Tags                     | EOL image tags produced using these pipelines                                                    | [Zenodo - EOL Computer Vision](https://zenodo.org/communities/eol/records?q=computer%20vision&l=list&p=1&s=10&sort=bestmatch)     |
| 🧠 Trained Models (Colab use)     | Models are set up for direct download within Colab notebooks                                     | _Embedded in Colab – no direct link_                                                                      |
| 🧪 Trained Models (Model Zoo)     | Models used for generating tags and crops                                                        | [EOL’s Kaggle Model Zoo](https://www.kaggle.com/eolorg/models)                                            |
| 🖼️ Training Images (Classes 1–4)  | Annotated training images for quality ratings (class 1–4, tagged by EOL users)                   | [Zenodo – Classes 1–4](https://zenodo.org/records/13305561)                                               |
| 🖼️ Training Images (Class 5)      | Exemplars for class 5 ("ideal" images)                                                           | [Zenodo – Class 5 Exemplars](https://zenodo.org/records/13305564)                                         |
|  **Object Detection for Image Cropping** |                                                                                                  |                                                                                                            |
| 🖼️ Ground Truth Images           | Ground truths vs. model-predicted boxes from `inspect_train_results.ipynb`                       | [Zenodo - Ground Truths vs Predictions](https://doi.org/10.5281/zenodo.14853913)                          |
| 🧠 Trained Models                | EOL-trained models used to generate tags and crops                                               | [Kaggle - EOL's Model Zoo](https://www.kaggle.com/eolorg/models)                                          |
| ✂️ Cropping Data                 | User-generated square cropping coordinates used to train models                                  | [Zenodo - User Cropping Data](https://zenodo.org/records/13305560)                                        |
|  **Object Detection for Image Tagging** |                                                                                                  |                                                                                                            |
| 🏷️ Image Tags                   | EOL image tags produced using object detection pipelines (upload in progress)                    | [Zenodo - EOL Computer Vision](https://zenodo.org/communities/eol/records?q=computer%20vision&l=list&p=1&s=10&sort=bestmatch)     |
|  **Other**                     |                                                                                                  |                                                                                                            |
| 💬 Feature Requests              | Request specific model files or datasets not listed above                                        | [Open an issue](https://github.com/aubricot/computer_vision_with_eol_images/issues/new/choose)           |


## References

<a id="note1" href="#note1ref"><sup>1</sup></a>[Ren et al. 2017](https://doi.org/10.1109/TPAMI.2016.2577031). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence.   
<a id="note2" href="#note2ref"><sup>2</sup></a>[Hui 2018](medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359). Object detection: speed and accuracy comparison (Faster R-CNN, R-FCN, SSD, FPN, RetinaNet and YOLOv3). Medium. 27 March 2018.   
<a id="note3" href="#note3ref"><sup>3</sup></a>[Redmon and Farhadi 2018](https://arxiv.org/pdf/1804.02767.pdf). YOLOv3: An Incremental Improvement.    
<a id="note4" href="#note4ref"><sup>4</sup></a>[Lin et al. 2015](https://arxiv.org/pdf/1405.0312.pdf). Microsoft COCO: Common Objects in Context.   
<a id="note5" href="#note5ref"><sup>5</sup></a>[Sandler et al. 2018](https://arxiv.org/abs/1801.04381). MobileNetV2: Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation. arXiv.   
<a id="note6" href="#note6ref"><sup>6</sup></a>[Szegedy et al. 2015](https://arxiv.org/abs/1512.00567). Rethinking the Inception Architecture for Computer Vision. arXiv.   
<a id="note7" href="#note7ref"><sup>7</sup></a>[Sharma 2019](medium.com/analytics-vidhya/image-classification-vs-object-detection-vs-image-segmentation-f36db85fe81). Image Classification vs. Object Detection vs. Image Segmentation. Medium. 23 Feb 2020.   
<a id="note8" href="#note8ref"><sup>8</sup></a>[Goeau et al. 2016](http://ceur-ws.org/Vol-1609/16090428.pdf). Plant identification in an open-world (LifeCLEF 2016). CEUR Workshop Proceedings.    
<a id="note9" href="note9ref"><sup>9</sup></a>[AlexeyAB 2020](https://github.com/AlexeyAB/darknet). Darknet. GitHub.   
<a id="note10" href="#note10ref"><sup>10</sup></a>[Bochkovskiy et al. 2020](https://arxiv.org/abs/2004.10934). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv.   
<a id="note11" href="#note11ref"><sup>11</sup></a>[Liu et al. 2016](https://doi.org/10.1007/978-3-319-46448-0_2). SSD: Single shot multibox detector. Lecture Notes in Computer Science.   
<a id="note12" href="#note12ref"><sup>12</sup></a>[Krasin et al. 2017](https://github.com/openimages). Open images: A public dataset for large-scale multi-label and multi-class image classification. GitHub.   


## License
**Code**  
Code in this repository is released under the [MIT license](https://github.com/aubricot/computer_vision_with_eol_images/blob/master/LICENSE). More information is available at the [Open Source Initiative](https://opensource.org/licenses/MIT).   
**Images**  
All images used in this repository and notebooks contained therein are licensed under [Creative Commons](https://creativecommons.org/licenses/). EOL content is freely available to the public. More information about re-use of content hosted by EOL is available at [EOL Terms of Use](https://eol.org/docs/what-is-eol/terms-of-use) and [EOL API Terms of Use](https://eol.org/docs/what-is-eol/terms-of-use-for-eol-application-programming-interfaces). Specific attribution information for EOL images used for training and testing models is available in bundle URLs containing "breakdown_download" found within notebooks.
