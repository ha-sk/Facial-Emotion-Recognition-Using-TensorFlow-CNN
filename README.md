# Facial-Emotion-Recognition-Using-TensorFlow-CNN
# 1. Introduction :

Emotions are highly valued because they play an important role in human
interaction. A model is created and trained to recognize real time emotions(Facial
Emotion Recognition) using Convolutional Neural Networks. The dataset is divided
into seven categories: neutral, fear, happy, sadness, surprise, disgust and anger.

# 2. Tools used:

## TensorFlow

TensorFlow is a free and open-source software library for dataflow and
differentiable programming across a range of tasks. It is a symbolic math library,
and is also used for machine learning applications such as neural networks. It is used
for both research and production at Google.

## Keras

Keras is an open-source neural-network library written in Python. It is capable of
running on top of TensorFlow, Microsoft Cognitive Toolkit, R, Theano, or PlaidML.
Designed to enable fast experimentation with deep neural networks, it focuses on
being user-friendly, modular, and extensible. It was developed as part of the
research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot
Operating System), and its primary author and maintainer is François Chollet, a
Google engineer. Chollet also is the author of the XCeption deep neural network
model.

Keras is based on minimal structure that provides a clean and easy way to create
deep learning models based on TensorFlow or Theano. Keras is designed to quickly
define deep learning models. Well, Keras is an optimal choice for deep learning
applications.

## NumPy

NumPy is a library for the Python programming language, adding support for
large, multi-dimensional arrays and matrices, along with a large collection of high-


level mathematical functions to operate on these arrays. The ancestor of NumPy,
Numeric, was originally created by Jim Hugunin with contributions from several
other developers. In 2005, Travis Oliphant created NumPy by incorporating
features of the competing Numarray into Numeric, with extensive modifications.
NumPy is open-source software and has many contributors.

NumPy targets the CPython reference implementation of Python, which is a non-
optimizing bytecode interpreter. Mathematical algorithms written for this version of
Python often run much slower than compiled equivalents. NumPy addresses the
slowness problem partly by providing multidimensional arrays and functions and
operators that operate efficiently on arrays, requiring rewriting some code, mostly
inner loops, using NumPy.

## Pandas

Pandas is a Python library for data analysis. Started by Wes McKinney in 2008
out of a need for a powerful and flexible quantitative analysis tool, pandas has
grown into one of the most popular Python libraries. It has an extremely active
community of contributors.

Pandas is built on top of two core Python libraries—matplotlib for data
visualization and NumPy for mathematical operations. Pandas acts as a wrapper
over these libraries, allowing you to access many of matplotlib's and NumPy's
methods with less code. For instance, pandas' .plot() combines multiple matplotlib
methods into a single method, enabling you to plot a chart in a few lines.

## Matplotlib

Matplotlib is a plotting library for the Python programming language and its
numerical mathematics extension NumPy. It provides an object-oriented API for
embedding plots into applications using general-purpose GUI toolkits like Tkinter,
wxPython, Qt, or GTK+. There is also a procedural "pylab" interface based on a
state machine (like OpenGL), designed to closely resemble that of MATLAB, though
its use is discouraged. SciPy makes use of Matplotlib. Several toolkits are available
which extend Matplotlib functionality. Some are separate downloads, others ship
with the Matplotlib source code but have external dependencies.

Pyplot is a Matplotlib module which provides a MATLAB-like interface. Matplotlib is
designed to be as usable as MATLAB, with the ability to use Python and the
advantage of being free and open-source. Each pyplot function makes some


change to a figure: e.g., creates a figure, creates a plotting area in a figure, plots
some lines in a plotting area, decorates the plot with labels, etc. The various plots
we can utilize using Pyplot are Line Plot, Histogram, Scatter, 3D Plot, Image,
Contour, and Polar.

## OpenCV:

OpenCV (Open Source Computer Vision Library) is a library of programming
functions mainly aimed at real-time computer vision. cv2 is the module import name
for opencv-python, "Unofficial pre-built CPU-only OpenCV packages for Python".

OpenCV provides a training method (see Cascade Classifier Training) or pretrained

models, that can be read using the cv::CascadeClassifier::load method. The

pretrained models are located in the data folder in the OpenCV installation or can
be found.

# 3. Dataset:

The dataset fer- 2013 which is an open-source data set that was made publicly
available for a Kaggle competition. It contains 48 X 48-pixel grayscale images of
the face. There are seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy,
4=Sad, 5=Surprise, 6=Neutral) present in the data. The CSV file contains two
columns that are emotion that contains numeric code from 0-6 and a pixel column
that includes a string surrounded in quotes for each image.

The dataset used can be downloaded from
https://www.kaggle.com/deadskull7/fer

# 4. Methodology:

The CNN model was created on Jupyter notebook. TensorFlow is used as the
machine learning framework. First, all of the required libraries are imported. The
dataset is then imported and assigned to the appropriate data object.


## a. Importing the libraries:

**import tensorflow as tf**

**import numpy as np**

**import cv**

**import matplotlib.pyplot as plt**

**import sys,os**

**import pandas as pd**

Further some modules from the Tensorflow library are also imported.

## b. Importing the Dataset

The Emotion Dataset is imported. After importing we have printed the data frame
as shown in the image.

## c. Data preprocessing

Data preprocessing is one of the important steps in the machine learning pipeline.

load the pixels CSV of the file into a data frame.

## d. Splitting Dataset

Split the data set into the train and validation set. So that we can check whether the
model is overfitted to the training dataset or not using the validation dataset.


## e. Building Facial Emotion Detection Model using CNN

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm that
can take in an input image, assign importance (learnable weights and biases) to
various aspects/objects in the image, and be able to differentiate one from the
other. The pre-processing required in a ConvNet is much lower as compared to
other classification algorithms. While in primitive methods filters are hand-
engineered, with enough training, ConvNets can learn these filters/characteristics.

A Convolution Neural Network or CNN as it is popularly called is a collection of
mainly two types of layers:

- The hidden layers / Feature Extraction Part
→ convolutions
→ pooling
- The classifier part
    1. We first need to initiate sequential class since there are various layers to
       build CNN which all must be in sequence. Then we add the first
       convolutional layer where we need to specify 5 arguments.
    2. We need to apply the pooling operation after initializing CNN. Pooling is
       an operation of down sampling of the image. The pooling layer is used to
       reduce the dimensions of the feature maps. Thus, the Pooling layer
       reduces the number of parameters to learn and reduces computation in
       the neural network.
    3. In order to add two more convolutional layers, we need to repeat steps
       1 & 2 with slight modification in the number of filters.
    4. Flattening operation is converting the dataset into a 1-D array for input
       into the next layer which is the fully connected layer.
       After finishing the 3 steps, now we have pooled feature map. We are
       now flattening our output after two steps into a column. Because we need
       to insert this 1-D data into an artificial neural network layer.
    5. The output of the flattening operation work as input for the neural
       network. The aim of the artificial neural network makes the convolutional
       neural network more advanced and capable enough of classifying
       images.
       Here we are using a dense class from the Keras library from creating a
       fully connected layer and output layer.


## f. Training the model

Now it’s time to train our model. Training is nothing but a learning loop. here we
define hyperparameters such as the number of epochs, batch size, learning rate
etc..The only way to find the best parameters is by trying.

we use callbacks to record model performance while the model is learning.

## g. Saving the model

Save the model, It involves both weights and architecture.

We don’t need to save the architecture but every time when we load model, we
have to again define model architecture before we use it.

So it’s preferable to save both (model into .h5 file and architecture into json file.)

# 5. Test

The test bellow will be a realtime webcam test showing the camera some picture
from Google because our faces are not well predicted.

1.This was supposed to show us disgust but it returns neutral.


2.The prediction was correct the child looks happy.

3.The prediction was correct the child looks sad.

4.The prediction is also correct because this child looks neutral.


5.The prediction is correct because the child looks angry.

6.The prediction was correct because the child was looking feared.

As we can see the program predicted 5 / 6 faces and get one incorrect answer.

# 6.Conclusion

After training the model, it wasn’t trained well but we could test it, and gave us
some accurate prediction.

For more accurate predictions, it is better to retrain it more by increasing the
number of epochs.


