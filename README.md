# Computer-vision

Computer Vision (CV) is a field that studies how computers can gain some degree of understanding from digital images and/or video.
The most common problems of computer vision include:
* Image classification
* Object Detection
* Segmentation

To deal with color images, we need some way to represent colors. In most cases, we represent each pixel by 3 intensity values, 
corresponding to Red (R), Green (G) and Blue (B) components. This color encoding is called RGB, and thus color image of size 
W×H will be represented as an array of size H×W×3 (sometimes the order of components might be different, but the idea is the same).
Multi-dimensional arrays are also called **tensors** (before training any models we need to convert our dataset into a set of tensors).

Use **Open CV, or PIL/Pillow, or imageio** to load your image into numpy array, you can easily convert it to tensors. It is important to make sure that all values are **scaled to the range [0..1]** before you pass them to a neural network - it is the usual convention for data preparation. It is important to note that all images should be scaled to the same **size** (either by cropping images, or by padding extra space). 

Data set used:
* MNIST
* Fashion MNIST
* CIFAR-10

### TensorFlow
In version 2 TensorFlow added a higher-level neural network construction API called Keras. With Keras, most model building steps can be done in a much simpler way. Only switching to pure TensorFlow when you need to develop some custom architectures for research or more complex scenarios.

### PyTorch
Even better approach is to use functionality in Torchvision library, namely ImageFolder. It does all the preprocessing steps automatically, and also assigns labels to images according to the directory structure

## Simple dense neural network (DNN)
Fully-connected layer or Dense layer:
* imge shape
* input layer (flatten)
* output layer (softmax)

In **Multi-layer network** take account:
* A number of parameters of nerual network should be chosen depending on the dataset size, to prevent overfitting.
* Inserting non-linear functions in between layers is important (ReLU), achieve high expressive power.

**Overfitting** is a very important concept to understand. It means that our model fits the training data very well, but does not further generalize well on unseen data. Often overfitting results in validation accuracy starting to increase, which means that model is becoming worse with further training. What you can do to overcome overfitting:
* Make the model less powerful by decreasing the number of parameters
* Increase the number of training examples, maybe by using some clever approaches such as data augmentation
* Stop training as soon as validation accuracy starts dropping

Multi-level networks can achieve higher accuracy than single-layer perceptron, however, they are not perfect for computer vision tasks. In images, there are some structural patterns that can help us classify an object regardless of it's position in the image, but perceptrons do not allow us to extract those patterns and look for them selectively.

## Convolutional Neural Networks (CNN)
## Pre-trained network with transfer learning
## MobileNet5

