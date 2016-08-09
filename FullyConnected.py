import numpy as np
from NeuralNet.classifiers.fc_net import *
from NeuralNet.data_utils import get_CIFAR10_data
from NeuralNet.solver import Solver

#load CIFAR 10 dataset
data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

"""
- Here [100, 100, 100, 100, 100] is the size of each hidden layer and the length of this list determines the size of the network 
- The FullyConnectedNet class is defined in NeuralNet/classifiers/fc_net.py 
- weight_Scale is used as standard deviation for random initailization of weights 
- Read NeuralNet.classifiers.fc_net for more information on the paramters used in the constructor
- Use 'sgd' for using Stochastic Gradient Descent, 'sgd_momentum' for Stochastic Gradient Descent + momentum update, 'rmsprop' for RMSprop and 'adam' for 'ADAM' optimization algorithm
"""

model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2, reg=0.0,dropout=0, use_batchnorm=False)

solver = Solver(model, data,
                num_epochs=20, batch_size=100,
                update_rule='adam',
                optim_config={
                'learning_rate': 1e-3
                },
                verbose=True)

solver.train()

y_test_pred = np.argmax(model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(model.loss(data['X_val']), axis=1)
print 'Validation set accuracy: ', (y_val_pred == data['y_val']).mean()
print 'Test set accuracy: ', (y_test_pred == data['y_test']).mean()
