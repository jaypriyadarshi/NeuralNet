import numpy as np 
from NeuralNet.classifiers.cnn import *
from NeuralNet.data_utils import get_CIFAR10_data
from NeuralNet.layers import *
from NeuralNet.fast_layers import *
from NeuralNet.solver import Solver
import pickle

#load CIFAR 10 dataset
data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

model = pickle.load(open("model", "rb"))

y_test_pred = np.argmax(model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(model.loss(data['X_val']), axis=1)
print 'Validation set accuracy: ', (y_val_pred == data['y_val']).mean()
print 'Test set accuracy: ', (y_test_pred == data['y_test']).mean()


