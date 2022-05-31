# Classification for Image Tagging
Testing different image classification frameworks as a method to automatically generate tags for flowers, image type (map/herbarium sheet/phylogeny/illustration), and image ratings within EOL images.   
*Last updated 31 May 2022*

## Project Structure
**Flowers**
* Pre-trained [MobileNet SSD v2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4) and [Inception v3](https://tfhub.dev/google/imagenet/inception_v3/classification/4) models downloaded from [Tensorflow Hub](https://www.tensorflow.org/hub) and a custom-built model made from scratch based on advice from the [Tensorflow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification) and [pyimagesearch](https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/) were trained on the [PlantCLEF 2016 Image dataset](https://www.imageclef.org/lifeclef/2016/plant) (Goeau et al. 2016) to classify images into flower, fruit, stem, branch, entire or leaf. Model results were inconsistent and a flower/fruit classifier was built for use instead.

:arrow_right: :seedling: Because trained model results were inconsistent, use flowers/fruits pipeline below to add flower tags to images.

**Flowers/Fruits**
* Pre-trained [MobileNet SSD v2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4) and [Inception v3](https://tfhub.dev/google/imagenet/inception_v3/classification/4) models downloaded from [Tensorflow Hub](https://www.tensorflow.org/hub) were trained on EOL Angiosperm images (evenly distributed across taxonomic groups) to classify images into flower/fruit or not flower/fruit. The model and hyperparameter combinations with the best trade-off between speed and accuracy - MobileNet SSD v2 training attempts 7 and 11 - were selected and used to generate image tags for a bundle of 20,000 EOL flowering plant images. For test images, training attempt 7 performed better at identifying reproductive structures (flower/fruit) and attemt 11 performed better at identifying images without reproductive structures. Confidence thresholds to maximize coverage and minimize error for both model's predictions were used to post-process and filter classification results and produce final tags.

:arrow_right: :seedling: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/object_detection_for_image_tagging/flower_fruit/classify_images.ipynb) Click here to start generating crops.

**Image Type**
* Pre-trained [MobileNet SSD v2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4) and [Inception v3](https://tfhub.dev/google/imagenet/inception_v3/classification/4) models downloaded from [Tensorflow Hub](https://www.tensorflow.org/hub) were trained on EOL, Wikimedia Commons, and Flicker BHL images to classify EOL images as a map, phylogeny, illustration, herbarium sheet, or none. The model and hyperparameter combinations with the best trade-off between speed and accuracy - MobileNet SSD v2 training attempt 13 - was selected and used to generate image tags for a bundle of 20,000 EOL images. Images are pre-processed to add photographic or non-photographic tags using a "cartoonization" image processing approach. Then image tags are added using image classification with MobileNet SSD v2. 

:arrow_right: :seedling: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/image_type/classify_images.ipynb) Click here to start generating crops.

**Image Ratings**
* Pre-trained [MobileNet SSD v2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4) and [Inception v3](https://tfhub.dev/google/imagenet/inception_v3/classification/4) models downloaded from [Tensorflow Hub](https://www.tensorflow.org/hub) were trained using EOL user generated image rating and exemplar datasets to and used to classify EOL images into quality rating categories 1-5 (worst to best). Training results were not consistent despite large datset sizes (7k images per class), likely attributed to the subjective nature of user quality ratings for what defines a "good" image. The model and hyperparameter combinations with the best trade-off between speed and accuracy - Inception v3 training attempt 20 - was selected and used to generate image tags. Results were consistent for "bad" images (ie users were more unified on what they don't like), so this was leveraged to keep model outputs for "bad" (classes 1-2) and tag remaining images as "good." 

:arrow_right: :seedling: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/rating/classify_images.ipynb) Click here to start generating crops.

## Getting Started  
Except preprocessing.py files (plantclef_preprocessing.py), all files in this repository are run in [Google Colab](https://research.google.com/colaboratory/faq.html) (Google Colaboratory, "a free cloud service, based on Jupyter Notebooks for machine-learning education and research"). Using Colab, everything is run entirely in the cloud (and can link to Google Drive) and no local software or library installs are requried.   

For additional details on steps below, see the [project wiki](https://github.com/aubricot/computer_vision_with_eol_images/wiki).

## Flowers
**Step 1) Pre-process training data locally and upload to Google Drive:** Download the [PlantCLEF 2016 Image dataset](https://www.imageclef.org/lifeclef/2016/plant) locally. Then run [plantclef_preprocessing.py](https://github.com/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/flowers/plantclef_preprocessing.py) to randomly select 6,000 images and sort them into folders based on image classes contained within metadata of the training dataset xmls (flower, fruit, entire, stem, leaf, branch). After running plantclef_preprocessing.py, zip the folder containing all image class subfolders. Upload the zipped folder to Google Drive (uploading the zipped folder will save time, because uploading to Drive can be slow) and unzip before running the notebook in Step 2 (use a command like this in a Colab notebook: !unzip images.zip -d images).

**Step 2) Build & train classification models:** After preparing the training dataset locally and uploading it to Google Drive in Step 1, build and train classification models by clicking here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/flowers/flowers_train.ipynb). Form fields and dropdown menus within the notebook walk you through model selection, adjusting hyperparameters, training, and the display of trained models.

**Step 3) Display classification results on images:** To display classification results from Step 2 on images and verify that they are as expected (or to further fine tune the classification model accordingly, ex: adjust hyperparameters from drop-down menus and re-train), click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/flowers/classification_display_test.ipynb).

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/images/classification_example.jpg" align="middle" width="300" ></a></p>   

<p align="center">
<sub><sup>Image is hosted by Encyclopedia of Life (<a href="https://content.eol.org/data/media/66/a1/2a/509.63397702.jpg"><i>Leucopogon tenuicaulis</i></a>, licensed under <a href="https://creativecommons.org/licenses/by/3.0/">CC BY 3.0</a>).</sup></sub>

## Flowers/Fruits
**Step 1) Pre-process training data in Google Drive:** Click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/flower_fruit/flower_fruit_preprocessing.ipynb) to download images from  EOL Angiosperm "max 30 images per family" bundle to Google Drive. Manually sort images into flower, fruit, null (no flowers or fruits), and other (images to be excluded that don't fit cleanly into oather categories) folders. Then, return to notebook to inspect the number of images per folder and training class. The number of training images per class should be even, so the folder with the fewest images (fruit) is used as is. Other folders (flower, null) are archived and zipped before trimming the number of images contained to be even across all training folders.

**Step 2) Build & train classification models:** After pre-processing the training dataset in Step 1, build and train classification models by clicking here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/flower_fruit/flower_fruit_train.ipynb). Form fields and dropdown menus within the notebook walk you through model selection, adjusting hyperparameters, training, and the display of trained models. Choose the N-best training attempts/models before moving to Step 3.

**Step 3) Determine model prediction confidence thresholds to use for classification:** Click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/flower_fruit/det_conf_threshold.ipynb) to review false and positive detections by class for attempts/models chosen in Step 2. Choose which trained model and confidence threshold values to use for classifying flowers/fruits from EOL images. Threshold values should be chosen that maximize coverage and minimize error.

**Step 4) Classify images, post-process results, and display outcomes:** To classify images, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/flower_fruit/classify_images.ipynb).
Run images through classification models selected in Step 2. Then post-process results using confidence values chosen in Step 3. Display final classification tags on images and verify that they are as expected (or further fine tune confidence thresholds).  

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/images/fruit_prediction.jpg" align="middle" width="300" ></a></p>   

<p align="center">
<sub><sup>Image is hosted by Encyclopedia of Life (<a href="https://content.eol.org/data/media/56/65/2f/509.15852359.jpg"><i>Curtisia assegai</i></a>, licensed under <a href="http://creativecommons.org/licenses/by-sa/3.0/">CC BY SA 3.0</a>).</sup></sub>

## Image Type
**Step 1) Pre-process training data in Google Drive:** Click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/image_type/image_type_preprocessing.ipynb) to download images from  map, phylogeny, illustration, and herbarium sheet image bundles to Google Drive. Manually inspect images to confirm they fit within the appropriate folders. Then, return to notebook to build null image class and inspect the number of images per folder and training class. The number of training images per class should be even, so the folder with the fewest images (phylogeny) is used as is. Other folders (map, herbarium sheet, illustration, null) are archived and zipped before trimming the number of images contained to be even across all training folders.

**Step 2) Build & train classification models:** After pre-processing the training dataset in Step 1, build and train classification models by clicking here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/image_type/image_type_train.ipynb). Form fields and dropdown menus within the notebook walk you through model selection, adjusting hyperparameters, training, and the display of trained models. Choose the N-best training attempts/models before moving to Step 3.

**Step 3) Determine model prediction confidence thresholds to use for classification:** Click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/image_type/inspect_train_results.ipynb) to review false and positive detections by class for attempts/models chosen in Step 2. Choose which trained model and confidence threshold values to use for classifying EOL images into different type categories. Threshold values should be chosen that maximize coverage and minimize error. 

**Step 4) Determine cartoonization values to use for classifying images as photographic or non-photographic:** Click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/image_type/cartoonify_images.ipynb) to determine cartoonization values for the training dataset. Images are "caroonified", then the change in color values is compared. If change above a certain threshold, then it is likely a photographic image. If change below a certain threshold, it is likely a non-photographic image (cartoon). Manhattan norm, Manhattan norm per pixel, Zero norm, and Zero norm per pixel are shown. Threshold values should be chosen that clearly distinguish the two categories for the training dataset. 

**Step 5) Cartoonize images, Classify images, post-process tags, and display results:** To classify images, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/image_type/classify_images.ipynb).
Run images through cartoonization and classification pipelines. Then post-process results using confidence values chosen in Step 3 and cartoonization values chosen in Step 4. Display final classification tags on images and verify that they are as expected (or further fine tune thresholds).  

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/images/cartoonization.jpg" align="middle" width="300" ></a></p>  

<p align="center">
<sub><sup>Images are hosted by Encyclopedia of Life (left: <a href="https://content.eol.org/data/media/9f/65/3c/600.22397755.98x68.jpg"><i>Magnolia sp.</i></a>, licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/">CC BY NC SA 3.0</a>, right: <a href="https://content.eol.org/data/media/99/27/2a/84.CalPhotos_4444_4444_0510_1371.98x68.jpg"><i>Passiflora sp.</i></a>, licensed under <a href="http://creativecommons.org/licenses/publicdomain/">Creative Commons Public Domain</a>).</sup></sub>

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/images/sample_map.jpg" align="middle" width="500" ></a></p>   

<p align="center">
<sub><sup>Image is hosted by Wikimedia Commons user: Ikonact (<a href="https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/New_York_state_geographic_map-en.svg/800px-New_York_state_geographic_map-en.svg.png">New York state geographic map-en.svg</a>, licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/deed.en">CC BY-SA 4.0</a>).</sup></sub>

## Image Ratings
**Step 1) Pre-process training data in Google Drive:** Click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/rating/rating_preprocessing.ipynb) to download images from  EOL user generated rating and exemplar datasets to their respective folders in Google Drive for training and testing classification models. 

**Step 2) Build & train classification models:** After pre-processing the training dataset in Step 1, build and train classification models by clicking here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/rating/rating_train.ipynb). Form fields and dropdown menus within the notebook walk you through model selection, adjusting hyperparameters, training, and the display of trained models. Choose the N-best training attempts/models before moving to Step 3.

**Step 3) Determine model prediction confidence thresholds to use for classification:** Click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/rating/inspect_train_results.ipynb) to review false and positive detections by class for attempts/models chosen in Step 2. Choose which trained model and confidence threshold values to use for classifying EOL images by rating. Threshold values should be chosen that maximize coverage and minimize error. Additional option to inspect results by taxon. 

Training results were not consistent for "good" images but were consistent for "bad" images despite large datset sizes (7k images per class). Due to the subjective nature of user quality ratings, we assume that users were conflicted on what makes an image "good," but unified on what makes an image "bad". The model and hyperparameter combinations with the best trade-off between speed and accuracy - Inception v3 training attempt 20 - was selected and used to generate image tags for sample datasets of 50 EOL images that were manually inspected.

**Step 4) Classify images, post-process results, and display outcomes:** To classify images, click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/rating/classify_images.ipynb).
Run images through classification models using confidence values for "bad" image predictions selected in Step 3. Display final classification tags on images and verify that they are as expected (or further fine tune confidence thresholds).  

<p align="center">
<a href="url"><img src="https://github.com/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/images/goodbad_rating_ex.jpg" align="middle" width="500" ></a></p>   

<p align="center">
<sub><sup>Are the shown images good or bad? EOL users rated the left image of a goat as bad (1) and the right image of a Collared Peccary as good (5). Images are hosted by Flickr (<a href="http://farm2.staticflickr.com/1421/5177003577_d5c66874a7_o.jpg"><i>Pecari tajacuas</i></a>, licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/2.0/">CC-BY-NC-SA-2.0</a>) and Wikimedia commons (<a href="https://commons.wikimedia.org/wiki/File:Nederlandse_witte_geit.jpg"><i>Capra sp.</i></a>, licensed under <a href="https://creativecommons.org/licenses/by-sa/3.0/deed.en">CC-BY-SA-3.0</a>).</sup></sub> 
  
## References
* [Çakır 2019](https://medium.com/analytics-vidhya/create-tensorflow-image-classification-model-with-your-own-dataset-in-google-colab-63e9d7853a3e). Create Tensorflow Image Classification Model with Your Own Dataset in Google Colab. Medium. 31 Oct 2019.
* [Chollet 2016](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html). Building powerful image classification models using very little data. The Keras Blog. 5 Jun 2016.
* [Custom training: walkthrough 2020](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough). Tensorflow Core. 12 Jun 2020.
* [Encyclopedia of Life](eol.org)
* [Goeau et al. 2016](http://ceur-ws.org/Vol-1609/16090428.pdf). Plant identification in an open-world (LifeCLEF 2016). CEUR Workshop Proceedings. 
* [Pegwar 2020](https://medium.com/analytics-vidhya/how-to-do-image-classification-on-custom-dataset-using-tensorflow-52309666498e). How to do Image Classification on custom Dataset using TensorFlow. Medium. 1 Apr 2020.
* [PlantCLEF 2016 Image dataset](https://www.imageclef.org/lifeclef/2016/plant)
* [Rosebrock 2018](https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/). Keras Conv2D and Convolutional Layers. pyimagesearch. 31 Dec 2018.
* [Sandler et al. 2018](https://arxiv.org/abs/1801.04381). Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation." arXiv.
* [Sharma 2019](medium.com/analytics-vidhya/image-classification-vs-object-detection-vs-image-segmentation-f36db85fe81). Image Classification vs. Object Detection vs. Image Segmentation. Medium. 23 Feb 2020.
* [Szegedy et al. 2015](https://arxiv.org/abs/1512.00567). Rethinking the Inception Architecture for Computer Vision. arXiv.
* [Tensorflow Image Classification Tutorial 2020](https://www.tensorflow.org/tutorials/images/classification). Tensorflow Core. 10 Jul 2020.
