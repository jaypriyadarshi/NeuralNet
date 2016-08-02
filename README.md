# NeuralNet
- A Python library to train and evaluate Neural Networks and Convolutional Neutral Networks
- Requires Numpy

Features:

- An optimized modular neural network library written in python
- Fully vectorized  implementation for efficient computation (using numpy)
- Layer types:  
  - Fully Connected
  - Dropout 
  - ReLU 
  - Convolutional layers
  - Max Pooling
- Normalization:
  - Vanilla Batch Normalization 
  - Spatial Batch Normalization for convolutional layers
- Optimization Algorithms:
  - Vanilla Stochastic Gradient Descent
  - Stochastic Gradient Descent + Momentum update
  - RMSprop
  - Adam
- Loss functions:
  - Softmax
  - SVM  

Loading CIFAR-10 dataset:
- cd NeuralNet/datasets
- ./get_datasets.sh

Data:
- Data is of the form (N, d1, d2, d3, ..., dn) where N is the #examples, d1, d2, d3, ..., dn is the dimentionality of the data
- Labels have (N) entries, where N is the #examples and each entry is the class of the corresponding example 
- eg: CIFAR-10 each image has 3 * 32 * 32 dimentions, so for 10000 examples the input takes the form (10000, 3, 32, 32) 

Examples:
- FullyConnected.py demonstrates how to train a Fully Connected Network
- convNet.py demonstrates how to train a seven layer convNet( uses batch normalization and dropout) - achieved 83.6% on CIFAR 10 in 20 epochs

Created using the code I wrote for the Stanford Course: CS231n: Convolutional Neural Networks for Visual Recognition 

