import numpy as np 
from NeuralNet.classifiers.cnn import *
from NeuralNet.data_utils import get_CIFAR10_data
from NeuralNet.layers import *
from NeuralNet.fast_layers import *
from NeuralNet.solver import Solver

#load CIFAR 10 dataset
data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

"""
- Training a Sever Layer ConvNet with dropout and batchnormalization layers, the class is defined in NeuralNet/classifiers/cnn.py
- To define your own architecture, you will need to write your own class like the ones in NeuralNet/classifiers/cnn.py
- weight_Scale is used as standard deviation for random initailization of weights 
- Read NeuralNet.classifiers.cnn for more information on the architecture and paramters used in the constructor
- Use 'sgd' for using Stochastic Gradient Descent, 'sgd_momentum' for Stochastic Gradient Descent + momentum update, 'rmsprop' for RMSprop and 'adam' for 'ADAM' optimization algorithm
"""

model = SevenLayerConvNetNorm(use_batchnorm=True, dropout=0.5, weight_scale=0.001, hidden_dim=1000, reg=0.001)

solver = Solver(model, data,
                num_epochs=40, batch_size=128,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()

y_test_pred = np.argmax(model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(model.loss(data['X_val']), axis=1)
print 'Validation set accuracy: ', (y_val_pred == data['y_val']).mean()
print 'Test set accuracy: ', (y_test_pred == data['y_test']).mean()
