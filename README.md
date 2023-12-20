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

**Tensors** are similar to NumPy arrays and ndarrays, except that tensors can run on GPUs or other hardware accelerators. In fact, tensors and NumPy arrays can often share the same underlying memory address with a capability called <code>bridge-to-np-label</code>, which eliminates the need to copy data. Tensors are also optimized for automatic differentiation 

**There are more than 100 tensor operations**, including arithmetic, linear algebra, matrix manipulation (such as transposing, indexing, and slicing). For sampling and reviewing, you'll find a comprehensive description [here](https://pytorch.org/docs/stable/torch.html). Each of these operations can be run on the GPU (at typically higher speeds than on a
CPU).
* CPUs have up to 16 cores. Cores are units that do the actual computation. Each core processes tasks in a sequential order (one task at a time).
* GPUs have 1000s of cores.  GPU cores handle computations in parallel processing. Tasks are divided and processed across the different cores. That's what makes GPUs faster than CPUs in most cases. GPUs perform better with large data than small data. GPU are typically used for high-intensive computation of graphics or neural networks (we'll learn more about that later in the Neural Network unit).
* PyTorch can use the Nvidia CUDA library to take advantage of their GPU cards.
By default, tensors are created on the CPU. Tensors can also be computed to GPUs; to do that, you need to move them using the `.to` method (after checking for GPU availability). Keep in mind that copying large tensors across devices can be expensive in terms of time and memory!

While training a model, we typically want to pass samples in "minibatches", reshuffle the data at every epoch to reduce model overfitting, and use Python's multiprocessing to speed up data retrieval.

Data set used:
* MNIST
* Fashion MNIST
* CIFAR-10

### TensorFlow
In version 2 TensorFlow added a higher-level neural network construction API called Keras. With Keras, most model building steps can be done in a much simpler way. Only switching to pure TensorFlow when you need to develop some custom architectures for research or more complex scenarios.

### PyTorch
The <code>torch.nn</code> namespace provides all the building blocks you'll need to build your own neural network. 

PyTorch provides two data primitives: <code>torch.utils.data.DataLoader</code> and <code>torch.utils.data.Dataset</code> that enable you to use pre-loaded datasets as well as your own data.

Even better approach is to use functionality in **Torchvision library**, namely <code>ImageFolder</code>. It does all the preprocessing steps automatically, and also assigns labels to images according to the directory structure. PyTorch domain libraries provide a number of sample pre-loaded datasets (such as FashionMNIST) that subclass torch.utils.data.Dataset and implement functions specific to the particular data.

## Deep neural network (DNN)
DA neural network is a collection of neurons that are connected by layers. Each neuron is a small computing unit that performs simple calculations to collectively solve a problem. Neurons are organized in 3 types of layers: **input layer, hidden layer, and output layer**. The hidden and output layers contain a number of neurons. Neural networks mimic the way a human brain processes information.
PUTIMAGE

### Components of neural network
* Activation function: determines whether a neruon should be activated or nor, if it is activate, then it means the input is important: *Binary,Siggmoid, Tanh, ReLU*
* Weights: influence how close the output of our network is to the expected output value. As an input enters the neuron, it gets multiplied by a weight value and the resulting output is either observed, or passed to the next layer in the neural network. Weights for all neurons in a layer are organized into one tensor
* Bias: makes up the difference between the activation function's output and its intended output. A low bias suggest that the network is making more assumptions about the form of the output, whereas a high bias value makes fewer assumptions about the form of the output.

$x = \sum{(weights * inputs) + bias} $, where $f(x)$ is the activation function.

### Observations
In **Multi-layer network** take account:
* A number of parameters of nerual network should be chosen depending on the dataset size, to prevent overfitting.
* Inserting non-linear functions in between layers is important (ReLU), achieve high expressive power.

**Overfitting** is a very important concept to understand. It means that our model fits the training data very well, but does not further generalize well on unseen data. Often overfitting results in validation accuracy starting to increase, which means that model is becoming worse with further training. What you can do to overcome overfitting:
* Make the model less powerful by decreasing the number of parameters
* Increase the number of training examples, maybe by using some clever approaches such as data augmentation
* Stop training as soon as validation accuracy starts dropping

Multi-level networks can achieve higher accuracy than single-layer perceptron, however, **they are not perfect for computer vision tasks**. In images, there are some structural patterns that can help us classify an object regardless of it's position in the image, but perceptrons do not allow us to extract those patterns and look for them selectively.

## Convolutional Neural Networks (CNN)
Computer vision is different from generic classification, because when we are trying to find a certain object in the picture, we are scanning the image looking for some specific **patterns** and their combinations. For example, when looking for a cat, we first may look for horizontal lines, which can form whiskers, and then certain combination of whiskers can tell us that it is actually a picture of a cat. The position and presence of certain patterns are important. 

key concepts in CNNs for computer vision: 

### Convolutional filter
IMG: https://medium.com/advanced-deep-learning/cnn-operation-with-2-kernels-resulting-in-2-feature-mapsunderstanding-the-convolutional-filter-c4aad26cf32
Convolutional filters are small windows that run over each pixel of the image and compute weighted average of the neighboring pixels. Example
$$
\left(
    \begin{matrix}
     -1 & 0 & 1 \cr
     -1 & 0 & 1 \cr
     -1 & 0 & 1 \cr
    \end{matrix}
\right)
$$
When this filter goes over relatively uniform pixel field, all values add up to 0. When it encounters a vertical edge in the image, high spiked value is generated. 

In classical computer vision, multiple filters were applied to the image to generate features, which then were used by machine learning algorithm to build a classifier. In deep learning we construct networks that learn the best convolutional filters to solve classification problem on its own.

### Convolutional layers
**TensroFlow**: Convolutional layers are defined using Conv2d class. We need to specify the following:
* filters - number of filters to use. We will use 9 different filters, which will give the network plenty of opportunities to explore which filters work best for our scenario.
* kernel_size is the size of the sliding window. Usually 3x3 or 5x5 filters are used.

### Pooling layers

## Pre-trained network with transfer learning
## MobileNet5

