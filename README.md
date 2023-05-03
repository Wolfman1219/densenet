# DenseNet Implementation with TensorFlow Keras

This repository contains an implementation of the DenseNet architecture using TensorFlow Keras. DenseNet is a deep convolutional neural network architecture designed for image classification tasks. It is known for its ability to mitigate the vanishing gradient problem by introducing skip connections, which allow the network to learn identity functions and improve gradient flow.

## DenseNet Architecture

DenseNet is a deep learning architecture that was introduced in 2016 by Huang et al. It is a convolutional neural network (CNN) that is designed to address the vanishing gradient problem that can occur in very deep networks.The architecture of DenseNet is based on the idea of densely connecting each layer to every other layer in a feed-forward fashion. This is achieved by concatenating the feature maps of all preceding layers and passing them as input to the current layer. This dense connectivity pattern allows for better feature reuse and gradient flow throughout the network, resulting in improved accuracy and reduced overfitting.
DenseNet is composed of several dense blocks, each of which consists of multiple layers. Within each dense block, the input to each layer is the concatenation of the feature maps from all preceding layers in the block. This dense connectivity pattern allows for a significant reduction in the number of parameters required to train the network, while still maintaining high accuracy. In addition to the dense blocks, DenseNet also includes transition layers, which are used to reduce the spatial dimensions of the feature maps and control the number of parameters in the network. These transition layers consist of a batch normalization layer, followed by a 1x1 convolutional layer and a 2x2 average pooling layer. Overall, DenseNet has shown to be highly effective in a variety of computer vision tasks, including image classification, object detection, and semantic segmentation. Its dense connectivity pattern and efficient use of parameters make it a powerful tool for deep learning practitioners.

## Getting Started

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```
2.Run DenseNet.py script with:
```bash
python DenseNet.py
```
