# NeuralNet
- A Python library to train and evaluate Neural Networks and Convolutional Neutral Networks
- Created using the code I wrote for Stanford Course: CS231n: Convolutional Neural Networks for Visual Recognition
-Requires Numpy

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
- Run:
- cd NeuralNet/datasets
- ./get_datasets.sh
