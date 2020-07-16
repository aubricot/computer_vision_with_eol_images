# Classification for Image Tagging
Testing different image classification frameworks as a method to automatically generate tags for flowers, maps/labels/illustrations, and image ratings within EOL images. Such tags will improve user experience by enabling search of information not included in metadata and further facilitate use of EOL’s less structured image content.
*Last updated 15 July 2020*

## Project Structure
* **Flowers**: Pre-trained [MobileNet SSD v2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4) and [Inception v3](https://tfhub.dev/google/imagenet/inception_v3/classification/4) models downloaded from [Tensorflow Hub](https://www.tensorflow.org/hub) and a custom-built model made from scratch based on advice from the [Tensorflow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification) and [pyimagesearch](https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/) were trained on the [PlantCLEF 2016 Image dataset](https://www.imageclef.org/lifeclef/2016/plant) to classify images into flower, fruit, stem, branch, entire or leaf. Classifications will be used to generate image tags to improve searchability of EOLv3 images. Tags will be added to classified images to improve searchability in EOLv3. Because the PlantCLEF 2016 dataset used includes other categories, they will be tested, but accuracy for the flower class is the most important tag to add for users and other classes may be dropped depending on training results. Pre-trained and custom-built models are trained on the [PlantCLEF 2016 Image dataset](https://www.imageclef.org/lifeclef/2016/plant) and used to classify EOL images of flowering plants (Angiosperms) of varying dimensions and quality. The model and hyperparameter combinations with the best trade-off between speed and accuracy - a custom-built model - was selected and used to generate image tags for a bundle of 20,000 EOL flowering plant images. Adding tags to images of flowers will allow users interested in phenology to search for flowers in bloom within EOL Angiosperm images.

* **Maps, Labels, and Illustrations**: [to be developed following an initial research phase on existing models and datasets] Pre-trained and custom-built models will be trained on a to be determined dataset and used to classify EOL images of maps, labels, and illustrations. This will further improve search functionality for users, allowing the sorting of taxon-specific images of individuals from images of distribution maps, collection labels, or illustrations.

* **Image Ratings**: [to be developed following an initial research phase on existing models and datasets] Pre-trained and custom-built models will be trained on a to be determined dataset (possibly from an existing EOL user-generated image quality ratings dataset) and used to classify EOL images into quality rating categories 1-5 (best to worst). In addition to previous EOL object detection for image thumbnail cropping efforts, this will further improve EOL gallery displays and functionality for API users, allowing the selection of only the best-rated images to be displayed or retrieved instead of a pool of mixed quality. 

Results from image classification models used for the different tasks can be used to inform future large-scale image processing and user features for EOLv3 images.

## Getting Started  
Except preprocessing.py files (plantclef_preprocessing.py, tbd_preprocessing.py), all files in this repository are run in [Google Colab](https://research.google.com/colaboratory/faq.html) (Google Colaboratory, "a free cloud service, based on Jupyter Notebooks for machine-learning education and research"). Using Colab, everything is run entirely in the cloud (and can link to Google Drive) and no local software or library installs are requried.   

[To be written after 20K flower tags bundle created]For additional details on steps below, see the [project wiki](https://github.com/aubricot/computer_vision_with_eol_images/wiki).

## Flowers
**Step 1) Pre-process training data locally and upload to Google Drive:** Download the [PlantCLEF 2016 Image dataset](https://www.imageclef.org/lifeclef/2016/plant) locally. Then run use [plantclef_preprocessing.py](https://github.com/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/flowers/plantclef_preprocessing.py) to randomly select 6,000 images and sort them into folders based on image classes contained within xmls of the training dataset (flower, fruit, entire, stem, leaf, branch). After running plantclef_preprocessing.py, zip the folder containing all image class subfolders. Upload the zipped folder to Google Drive (uploading the zipped folder will save time, because uploading to Drive can be slow) and unzip before running the notebook in Step 2 (use a command like this in a Colab notebook: !unzip images.zip -d images).

**Step 2) Build & train classification models:** After preparing the training dataset locally and uploading it to Google Drive in Step 1, build and train classification models by clicking here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/flowers/flowers_train.ipynb). Form fields and dropdown menus within the notebook walk you through model selection, adjusting hyperparameters, training, and the display of trained models.

**Step 3) Display classification results on images:** To display classification results from Step 2 on images and verify that they are as expected (or to further fine tune the classification model accordingly, ex: adjust hyperparameters from drop-down menus and re-train), click here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/computer_vision_with_eol_images/blob/master/classification_for_image_tagging/classification_display_test.ipynb).

## Maps, Labels, and Illustrations
[to be written following an initial research phase on existing models and datasets]

## Image Ratings
[to be written following an initial research phase on existing models and datasets]

## References
* https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
* https://www.tensorflow.org/tutorials/images/classification
* https://medium.com/analytics-vidhya/create-tensorflow-image-classification-model-with-your-own-dataset-in-google-colab-63e9d7853a3e
* https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb#scrollTo=umB5tswsfTEQ
* https://medium.com/analytics-vidhya/how-to-do-image-classification-on-custom-dataset-using-tensorflow-52309666498e
* https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
* https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
