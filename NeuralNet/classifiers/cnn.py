import numpy as np

from NeuralNet.layers import *
from NeuralNet.fast_layers import *
from NeuralNet.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    C, H, W = input_dim
    self.params['W1'] = np.random.normal(0,weight_scale,(num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)
    # after max pooling the size will be reduced by 2 along the width and 2 along the height
    self.params['W2'] = np.random.normal(0,weight_scale,(H * W * num_filters / 4, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(0,weight_scale,(hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)


    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    hidden_out, hidden_cache = affine_relu_forward(conv_out, W2, b2)
    scores, out_cache = affine_forward(hidden_out, W3, b3)
    
    if y is None:
      return scores
    
    loss, grads = 0, {}

    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    loss = data_loss + reg_loss

    dhidden, dW3, db3 = affine_backward(dscores, out_cache)
    dconv_out, dW2, db2 = affine_relu_backward(dhidden, hidden_cache)
    dx, dW1, db1 = conv_relu_pool_backward(dconv_out, conv_cache) 

    #regularization
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
    
    return loss, grads


class SevenLayerConvNet(object):
  """
  A seven-layer convolutional network with the following architecture:
  
  [conv - relu - conv - relu - 2x2 max pool]x2 - affine - relu - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=64, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    C, H, W = input_dim
    #conv_relu
    self.params['W1'] = np.random.normal(0,weight_scale,(num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)

    #conv_relu_pool
    self.params['W2'] = np.random.normal(0,weight_scale,(num_filters, num_filters, filter_size, filter_size))
    self.params['b2'] = np.zeros(num_filters)

    #conv_relu
    self.params['W3'] = np.random.normal(0,weight_scale,(num_filters, num_filters, filter_size, filter_size))
    self.params['b3'] = np.zeros(num_filters)

    #conv_relu_pool
    self.params['W4'] = np.random.normal(0,weight_scale,(num_filters, num_filters, filter_size, filter_size))
    self.params['b4'] = np.zeros(num_filters)

    #hidden-1 (as we have performed pooling twice)
    self.params['W5'] = np.random.normal(0,weight_scale,(H * W * num_filters / 16, hidden_dim))
    self.params['b5'] = np.zeros(hidden_dim)

    #hidden-2
    self.params['W6'] = np.random.normal(0,weight_scale,(hidden_dim, hidden_dim))
    self.params['b6'] = np.zeros(hidden_dim)

    #hidden-3
    self.params['W7'] = np.random.normal(0,weight_scale,(hidden_dim, num_classes))
    self.params['b7'] = np.zeros(num_classes)


    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    W7, b7 = self.params['W7'], self.params['b7']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    conv1_out, conv1_cache = conv_relu_forward(X, W1, b1, conv_param)
    conv2_out, conv2_cache = conv_relu_pool_forward(conv1_out, W2, b2, conv_param, pool_param)
    conv3_out, conv3_cache = conv_relu_forward(conv2_out, W3, b3, conv_param)
    conv4_out, conv4_cache = conv_relu_pool_forward(conv3_out, W4, b4, conv_param, pool_param)
    hidden1_out, hidden1_cache = affine_relu_forward(conv4_out, W5, b5)
    hidden2_out, hidden2_cache = affine_relu_forward(hidden1_out, W6, b6)
    scores, out_cache = affine_forward(hidden2_out, W7, b7)
    
    if y is None:
      return scores
    
    loss, grads = 0, {}

    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3) + np.sum(W4 * W4) + np.sum(W5 * W5) + np.sum(W6 * W6) + np.sum(W7 * W7))
    loss = data_loss + reg_loss

    dhidden2, dW7, db7 = affine_backward(dscores, out_cache)
    dhidden1, dW6, db6 = affine_relu_backward(dhidden2, hidden2_cache)
    dconv4, dW5, db5 = affine_relu_backward(dhidden1, hidden1_cache)
    dconv3, dW4, db4 = conv_relu_pool_backward(dconv4, conv4_cache)
    dconv2, dW3, db3 = conv_relu_backward(dconv3, conv3_cache)
    dconv1, dW2, db2 = conv_relu_pool_backward(dconv2, conv2_cache)
    dx, dW1, db1 = conv_relu_backward(dconv1, conv1_cache) 

    #regularization
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4
    dW5 += self.reg * W5
    dW6 += self.reg * W6
    dW7 += self.reg * W7

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'W5': dW5, 'b5': db5, 'W6': dW6, 'b6': db6, 'W7': dW7, 'b7': db7}

    
    return loss, grads

class SevenLayerConvNetNorm(object):
  """
  A seven-layer convolutional network with the following architecture:
  
  [conv - batchnorm - relu - conv - batchnorm - relu - 2x2 max pool]x2 - affine - batchnorm - relu - dropout - affine - batchnorm - relu - dropout - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=64, filter_size=3,
               hidden_dim=100, num_classes=10, dropout=0, use_batchnorm=False, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, seed=None):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    
    C, H, W = input_dim
    #conv_relu
    self.params['W1'] = np.random.normal(0,weight_scale,(num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)

    #conv_relu_pool
    self.params['W2'] = np.random.normal(0,weight_scale,(num_filters, num_filters, filter_size, filter_size))
    self.params['b2'] = np.zeros(num_filters)

    #conv_relu
    self.params['W3'] = np.random.normal(0,weight_scale,(num_filters, num_filters, filter_size, filter_size))
    self.params['b3'] = np.zeros(num_filters)

    #conv_relu_pool
    self.params['W4'] = np.random.normal(0,weight_scale,(num_filters, num_filters, filter_size, filter_size))
    self.params['b4'] = np.zeros(num_filters)

    #hidden-1 (as we have performed pooling twice)
    self.params['W5'] = np.random.normal(0,weight_scale,(H * W * num_filters / 16, hidden_dim))
    self.params['b5'] = np.zeros(hidden_dim)

    #hidden-2
    self.params['W6'] = np.random.normal(0,weight_scale,(hidden_dim, hidden_dim))
    self.params['b6'] = np.zeros(hidden_dim)

    #hidden-3
    self.params['W7'] = np.random.normal(0,weight_scale,(hidden_dim, num_classes))
    self.params['b7'] = np.zeros(num_classes)

    self.dropout_param = {}
    if self.use_dropout:
        self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed

    self.bn_params = []
    if self.use_batchnorm:
        self.bn_params = [{'mode': 'train', 'running_mean': np.zeros(num_filters, dtype=self.dtype), 'running_var': np.zeros(num_filters, dtype=self.dtype)},
        {'mode': 'train', 'running_mean': np.zeros(num_filters, dtype=self.dtype), 'running_var': np.zeros(num_filters, dtype=self.dtype)},
        {'mode': 'train', 'running_mean': np.zeros(num_filters, dtype=self.dtype), 'running_var': np.zeros(num_filters, dtype=self.dtype)},
        {'mode': 'train', 'running_mean': np.zeros(num_filters, dtype=self.dtype), 'running_var': np.zeros(num_filters, dtype=self.dtype)},
        {'mode': 'train', 'running_mean': np.zeros(hidden_dim, dtype=self.dtype), 'running_var': np.zeros(hidden_dim, dtype=self.dtype)},
        {'mode': 'train', 'running_mean': np.zeros(hidden_dim, dtype=self.dtype), 'running_var': np.zeros(hidden_dim, dtype=self.dtype)}
        ]
        all_gammas = {'gamma1': np.ones(num_filters), 'gamma2': np.ones(num_filters), 'gamma3': np.ones(num_filters), 'gamma4': np.ones(num_filters), 'gamma5': np.ones(hidden_dim), 'gamma6': np.ones(hidden_dim)}
        all_betas = {'beta1': np.zeros(num_filters), 'beta2': np.zeros(num_filters), 'beta3': np.zeros(num_filters), 'beta4': np.zeros(num_filters), 'beta5': np.zeros(hidden_dim), 'beta6': np.zeros(hidden_dim)}
        self.params.update(all_gammas)
        self.params.update(all_betas)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'

    if self.dropout_param is not None:
        self.dropout_param['mode'] = mode
    if self.use_batchnorm:
        for bn_param in self.bn_params:
            bn_param[mode] = mode

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    W7, b7 = self.params['W7'], self.params['b7']

    if self.use_batchnorm:
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        gamma3, beta3 = self.params['gamma3'], self.params['beta3']
        gamma4, beta4 = self.params['gamma4'], self.params['beta4']
        gamma5, beta5 = self.params['gamma5'], self.params['beta5']
        gamma6, beta6 = self.params['gamma6'], self.params['beta6']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    if self.use_batchnorm:
        conv1_out, conv1_cache = conv_batchnorm_relu_forward(X, W1, b1, conv_param, gamma1, beta1, self.bn_params[0])
        conv2_out, conv2_cache = conv_batchnorm_relu_pool_forward(conv1_out, W2, b2, conv_param, pool_param, gamma2, beta2, self.bn_params[1])
        conv3_out, conv3_cache = conv_batchnorm_relu_forward(conv2_out, W3, b3, conv_param, gamma3, beta3, self.bn_params[2] )
        conv4_out, conv4_cache = conv_batchnorm_relu_pool_forward(conv3_out, W4, b4, conv_param, pool_param, gamma4, beta4, self.bn_params[3])
        hidden1_out, hidden1_cache = affine_batchnorm_relu_forward(conv4_out, W5, b5, gamma5, beta5, self.bn_params[4])
        if self.use_dropout:
            hidden1_out, drop1_cache = dropout_forward(hidden1_out, self.dropout_param) 
        hidden2_out, hidden2_cache = affine_batchnorm_relu_forward(hidden1_out, W6, b6, gamma6, beta6, self.bn_params[5])
        if self.use_dropout:
            hidden2_out, drop2_cache = dropout_forward(hidden2_out, self.dropout_param)
        scores, out_cache = affine_forward(hidden2_out, W7, b7)

    else:
        conv1_out, conv1_cache = conv_relu_forward(X, W1, b1, conv_param)
        conv2_out, conv2_cache = conv_relu_pool_forward(conv1_out, W2, b2, conv_param, pool_param)
        conv3_out, conv3_cache = conv_relu_forward(conv2_out, W3, b3, conv_param)
        conv4_out, conv4_cache = conv_relu_pool_forward(conv3_out, W4, b4, conv_param, pool_param)
        hidden1_out, hidden1_cache = affine_batchnorm_relu_forward(conv4_out, W5, b5)
        hidden2_out, hidden2_cache = affine_batchnorm_relu_forward(hidden1_out, W6, b6)
        scores, out_cache = affine_forward(hidden2_out, W7, b7)
    
    if y is None:
      return scores
    
    loss, grads = 0, {}

    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3) + np.sum(W4 * W4) + np.sum(W5 * W5) + np.sum(W6 * W6) + np.sum(W7 * W7))
    loss = data_loss + reg_loss

    dhidden2, dW7, db7 = affine_backward(dscores, out_cache)
    if self.use_dropout:
        dhidden2 = dropout_backward(dhidden2, drop2_cache)
    dhidden1, dW6, db6, dgamma6, dbeta6 = affine_batchnorm_relu_backward(dhidden2, hidden2_cache)
    if self.use_dropout:
        dhidden1 = dropout_backward(dhidden1, drop1_cache)
    dconv4, dW5, db5, dgamma5, dbeta5 = affine_batchnorm_relu_backward(dhidden1, hidden1_cache)
    dconv3, dW4, db4, dgamma4, dbeta4 = conv_batchnorm_relu_pool_backward(dconv4, conv4_cache)
    dconv2, dW3, db3, dgamma3, dbeta3 = conv_batchnorm_relu_backward(dconv3, conv3_cache)
    dconv1, dW2, db2, dgamma2, dbeta2 = conv_batchnorm_relu_pool_backward(dconv2, conv2_cache)
    dx, dW1, db1, dgamma1, dbeta1 = conv_batchnorm_relu_backward(dconv1, conv1_cache) 

    #regularization
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4
    dW5 += self.reg * W5
    dW6 += self.reg * W6
    dW7 += self.reg * W7

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'W5': dW5, 'b5': db5, 'W6': dW6, 'b6': db6, 'W7': dW7, 'b7': db7, 'gamma1': dgamma1, 'beta1': dbeta1, 'gamma2': dgamma2, 'beta2': dbeta2, 'gamma3': dgamma3, 'beta3': dbeta3, 'gamma4': dgamma4, 'beta4': dbeta4, 'gamma5': dgamma5, 'beta5': dbeta5, 'gamma6': dgamma6, 'beta6': dbeta6}

    
    return loss, grads
  

class TenLayerConvNet(object):
  """
  A Ten-layer convolutional network with the following architecture:
  
  [conv - relu - conv - relu - 2x2 max pool]x3 - affine - relu - affine - relu - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=64, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    C, H, W = input_dim
    #conv_relu
    self.params['W1'] = np.random.normal(0,weight_scale,(num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)

    #conv_relu_pool
    self.params['W2'] = np.random.normal(0,weight_scale,(num_filters, num_filters, filter_size, filter_size))
    self.params['b2'] = np.zeros(num_filters)

    #conv_relu
    self.params['W3'] = np.random.normal(0,weight_scale,(num_filters, num_filters, filter_size, filter_size))
    self.params['b3'] = np.zeros(num_filters)

    #conv_relu_pool
    self.params['W4'] = np.random.normal(0,weight_scale,(num_filters, num_filters, filter_size, filter_size))
    self.params['b4'] = np.zeros(num_filters)

    #conv_relu
    self.params['W5'] = np.random.normal(0,weight_scale,(num_filters, num_filters, filter_size, filter_size))
    self.params['b5'] = np.zeros(num_filters)

    #conv_relu_pool
    self.params['W6'] = np.random.normal(0,weight_scale,(num_filters, num_filters, filter_size, filter_size))
    self.params['b6'] = np.zeros(num_filters)

    #hidden-1 (as we have performed pooling twice)
    self.params['W7'] = np.random.normal(0,weight_scale,(H * W * num_filters / 64, hidden_dim))
    self.params['b7'] = np.zeros(hidden_dim)

    #hidden-2
    self.params['W8'] = np.random.normal(0,weight_scale,(hidden_dim, hidden_dim))
    self.params['b8'] = np.zeros(hidden_dim)

    #hidden-3
    self.params['W9'] = np.random.normal(0,weight_scale,(hidden_dim, hidden_dim))
    self.params['b9'] = np.zeros(hidden_dim)

    #final
    self.params['W10'] = np.random.normal(0,weight_scale,(hidden_dim, num_classes))
    self.params['b10'] = np.zeros(num_classes)


    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    W7, b7 = self.params['W7'], self.params['b7']
    W8, b8 = self.params['W8'], self.params['b8']
    W9, b9 = self.params['W9'], self.params['b9']
    W10, b10 = self.params['W10'], self.params['b10']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    conv1_out, conv1_cache = conv_relu_forward(X, W1, b1, conv_param)
    conv2_out, conv2_cache = conv_relu_pool_forward(conv1_out, W2, b2, conv_param, pool_param)
    conv3_out, conv3_cache = conv_relu_forward(conv2_out, W3, b3, conv_param)
    conv4_out, conv4_cache = conv_relu_pool_forward(conv3_out, W4, b4, conv_param, pool_param)
    conv5_out, conv5_cache = conv_relu_forward(conv4_out, W5, b5, conv_param)
    conv6_out, conv6_cache = conv_relu_pool_forward(conv5_out, W6, b6, conv_param, pool_param)
    hidden1_out, hidden1_cache = affine_relu_forward(conv6_out, W7, b7)
    hidden2_out, hidden2_cache = affine_relu_forward(hidden1_out, W8, b8)
    hidden3_out, hidden3_cache = affine_relu_forward(hidden2_out, W9, b9)
    scores, out_cache = affine_forward(hidden3_out, W10, b10)

    
    if y is None:
      return scores
    
    loss, grads = 0, {}

    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3) + np.sum(W4 * W4) + np.sum(W5 * W5) + np.sum(W6 * W6) + np.sum(W7 * W7) + np.sum(W8 * W8) + np.sum(W9 * W9) + np.sum(W10 * W10))
    loss = data_loss + reg_loss

    #backprop
    dhidden3, dW10, db10 = affine_backward(dscores, out_cache)
    dhidden2, dW9, db9 = affine_relu_backward(dhidden3, hidden3_cache)
    dhidden1, dW8, db8 = affine_relu_backward(dhidden2, hidden2_cache)
    dconv6, dW7, db7 = affine_relu_backward(dhidden1, hidden1_cache)
    dconv5, dW6, db6 = conv_relu_pool_backward(dconv6, conv6_cache)
    dconv4, dW5, db5 = conv_relu_backward(dconv5, conv5_cache)
    dconv3, dW4, db4 = conv_relu_pool_backward(dconv4, conv4_cache)
    dconv2, dW3, db3 = conv_relu_backward(dconv3, conv3_cache)
    dconv1, dW2, db2 = conv_relu_pool_backward(dconv2, conv2_cache)
    dx, dW1, db1 = conv_relu_backward(dconv1, conv1_cache) 

    #regularization
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4
    dW5 += self.reg * W5
    dW6 += self.reg * W6
    dW7 += self.reg * W7
    dW8 += self.reg * W8
    dW9 += self.reg * W9
    dW10 += self.reg * W10

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'W5': dW5, 'b5': db5, 'W6': dW6, 'b6': db6, 'W7': dW7, 'b7': db7, 'W8': dW8, 'b8': db8, 'W9': dW9, 'b9': db9, 'W10': dW10, 'b10': db10}
   
    return loss, grads
  
pass
