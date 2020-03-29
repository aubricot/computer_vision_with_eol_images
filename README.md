# Computer Vision with EOL v3 Images 
Testing different computer vision methods (object detection, image classification) to do customized, large-scale image processing for [Encyclopedia of Life v3 database images](https://eol.org/pages/2913056/media).   
*Last updated 16 March 2020*

The Encyclopedia of Life (EOL) is an online biodiversity resource that seeks to provide information about all ~1.9 million species known to science. A goal for the latest version of EOL (v3) is to better leverage the older, less structured image content. To improve discoverability and display of EOL images, automated image processing pipelines that use computer vision with the goal of being scalable to millions of diverse images are being developed and tested.   

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/images/banner.jpg" align="middle" width="700" ></a></p>   

<sub><sup>Images a-c are hosted on Encyclopedia of Life (EOL) and licensed under Creative Commons (a. http://upload.wikimedia.org/wikipedia/commons/a/af/Choeronycteris_mexicana%2C_Mexican_long-tongued_bat_%287371567444%29.jpg , b. http://www.biolib.cz/IMG/GAL/34079.jpg, c. https://content.eol.org/data/media/7e/b3/54/542.16276541578.jpg).</sup></sub>

## Project Structure
* **Object detection for image cropping**: Three object detection frameworks (Faster-RCNN and SSD detection via the Tensorflow Object Detection API and YOLO via Darkflow) were used to perform customized, large-scale image processing for different groups of animals (birds, bats, butterflies & moths) found in the Encyclopedia of Life v3 database. The three frameworks differ in their speeds and accuracy: YOLO has been found to be the fastest but least accurate, while Faster RCNN was found to be the slowest but most accurate, with MobileNet SSD falling somewhere in between (Lin et al. 2017, Hui 2018, Redmon and Farhadi 2018). The model with the best trade-off between speed and accuracy for each group was selected to generate final cropping data for EOL images. The location and dimensions of the detected animals within each framework are used to crop images to square dimensions that are centered and padded around the detection box. For birds, pre-trained object detection models were used. For bats, butterflies and moths, object detection models were custom-trained using EOL user-generated cropping data (square coordinates around animal(s) of interest within each photo). [A model able to detect multiple taxonomic groups simultaneously is currently being developed and custom-trained].
<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_cropping/images/detected_bat.jpg" align="middle" width="700" ></a></p>   

* **Classification for image tagging**: [Future steps, not yet in progress] Image tagging will further improve EOL user experience, allowing users to search for features within images that are not already noted in metadata. Image classification can be used to label images with flowers present, images of maps, collection labels, and illustrations, and to generate image quality ratings. EOL user-generated cropping and image quality datasets will be used to custom train image classifiers. 

Results from object detection and image classification tasks will be used to inform future computer vision efforts and large-scale image processing for EOLv3 images. 

## Getting Started  
Except for .py files, all files in this repository are run in [Google Colab](https://research.google.com/colaboratory/faq.html) (Google Colaboratory, "a free cloud service, based on Jupyter Notebooks for machine-learning education and research"). Using Colab, everything runs entirely in the cloud (and can link to Google Drive) and no local software or library installs are requried. If running locally and using a GPU, there are several softwares that need to be installed first and take up ~10 GB and a few workarounds are required if running on a Windows OS. Working in the cloud eliminates these problems and makes it easier to collaborate if multiple users are on different operating systems. To use your local machine for object detection, refer to the [Tensorflow Object Detection API Tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb).

---   

## References
* Hui 2018. Object detection: speed and accuracy comparison (Faster R-CNN, R-FCN, SSD, 
FPN, RetinaNet and YOLOv3). Medium. 27 March 2018. [medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359](medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359).   
* Lin et al. 2015. Microsoft COCO: Common Objects in Context. [arXiv:1405.0312](https://arxiv.org/pdf/1405.0312.pdf).   
* Redmon and Farhadi 2018. YOLOv3: An Incremental Improvement. [arXiv:1804.02767](https://arxiv.org/pdf/1804.02767.pdf).    
