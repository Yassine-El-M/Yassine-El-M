
# Supervised Learning
## Project: Finding Donors for CharityML

### Install:

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Tensorflow](https://www.tensorflow.org/)
- [Tensorflow-hub](https://www.tensorflow.org/hub)

## Overview:

This project aims to classify images of flowers into one of 102 categories using the Oxford 102 flower dataset. The dataset includes a total of 8189 images of flowers, each belonging to one of 102 categories. The classifier was built using a convolutional neural network (CNN) architecture and was trained and tested using the Pytorch library.

## Dataset:

The Oxford 102 flower dataset is a widely used dataset for flower image classification, consisting of 102 categories of flowers with a total of 8189 images. The images were collected from a variety of sources and were annotated with labels indicating the species of the flower depicted in the image. The images were also resized and standardized to ensure consistent input to the classifier.

## Training and Evaluation:

The MobileNet pre-trained network was loaded from TensorFlow Hub. A new, untrained feed-forward network was defined as a classifier, which was trained and the loss and accuracy values were plotted during the training process for the training and validation sets. The final trained model was saved as a Keras model for command line application usage.

The classifier was trained using a combination of the Adam optimizer and the cross-entropy loss function. The model was trained for several epochs until the accuracy on the validation set reached a satisfactory level. The final model achieved an accuracy of 89% on the test set.

## Conclusion:

The image classifier built in this project was able to successfully classify images of flowers into one of 102 categories with an accuracy of 85%. This demonstrates the effectiveness of CNNs in image classification tasks and the utility of the Oxford 102 flower dataset in training and evaluating image classifiers.
