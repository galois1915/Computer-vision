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

Tools:
* Pytorch: ImageFolder does all the preprocessing steps automatically, and also assigns labels to images according to the directory structure. 

Data set used:
* MNIST
* CIFAR-10

### PyTorch
## Simple dense neural network
## Convolutional Neural Networks (CNN)
## Pre-trained network with transfer learning
## MobileNet5

