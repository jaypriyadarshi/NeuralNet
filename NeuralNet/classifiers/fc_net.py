import numpy as np

from NeuralNet.layers import *
from NeuralNet.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    self.params['W1'] = np.random.normal(0,weight_scale,(input_dim, hidden_dim))
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = np.random.normal(0,weight_scale,(hidden_dim, num_classes))
    self.params['b2'] = np.zeros(num_classes)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    out1, cache1 = affine_relu_forward(X, W1, b1)
    scores, cache2 = affine_forward(out1, W2, b2)

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}

    N = X.shape[0]
    data_loss = np.sum(-scores[range(N),y] + np.log(np.sum(np.exp(scores), axis=1)))
    data_loss /= N
    loss = data_loss + 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    #gradient on scores
    probs = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(N,1)
    dscores = probs
    dscores[range(N),y] -= 1
    dscores /= N

    #backprop through 2nd layer and 1st layer
    dhidden, dW2, db2 = affine_backward(dscores, cache2)
    dx, dW1, db1 = affine_relu_backward(dhidden, cache1) 

    dW1 += self.reg * W1
    dW2 += self.reg * W2

    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    self.params['W1'] = np.random.normal(0,weight_scale,(input_dim, hidden_dims[0]))
    self.params['b1'] = np.zeros(hidden_dims[0])
    for i in range(self.num_layers-2):
      self.params['W' + str(i + 2)] = np.random.normal(0,weight_scale,(hidden_dims[i], hidden_dims[i+1]))
      self.params['b' + str(i + 2)] = np.zeros(hidden_dims[i+1])
    self.params['W' + str(self.num_layers)] = np.random.normal(0,weight_scale,(hidden_dims[len(hidden_dims)-1], num_classes))
    self.params['b' + str(self.num_layers)] = np.zeros(num_classes)


    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train', 'running_mean': np.zeros(hidden_dims[i]), 'running_var': np.zeros(hidden_dims[i])} for i in xrange(self.num_layers - 1)]

      all_gammas = {'gamma' + str(i + 1): np.ones(hidden_dims[i]) for i in range(self.num_layers - 1)}
      all_betas = {'beta' + str(i + 1): np.zeros(hidden_dims[i]) for i in range(self.num_layers - 1)}
      
      self.params.update(all_gammas)
      self.params.update(all_betas)
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    
    out = X.copy()
    caches = []
    dropout_caches = []
    for i in range(self.num_layers-1):
      if self.use_batchnorm:
        W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]
        gamma = self.params['gamma' + str(i + 1)]
        beta = self.params['beta' + str(i + 1)]
        out, temp = affine_batchnorm_relu_forward(out, W, b, gamma, beta, self.bn_params[i])
        caches.append(temp)
      else: 
        W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]
        #temp variables stores cache and appends it to the caches list
        out, temp = affine_relu_forward(out, W, b)
        caches.append(temp)

      if self.use_dropout:
        out, drop_cache = dropout_forward(out, self.dropout_param)
        dropout_caches.append(drop_cache)

    #final forward pass
    W, b = self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)]
    scores, temp = affine_forward(out, W, b)
    caches.append(temp)


    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}

    data_loss, dscores = softmax_loss(scores, y)
    #regularization loss
    reg_loss = 0.0
    for i in range(self.num_layers):
      W = self.params['W' + str(i + 1)]
      reg_loss += np.sum(W * W)

    loss = data_loss + 0.5 * self.reg * reg_loss

    #backprop through last hidden layer
    W = self.params['W' + str(self.num_layers)]
    dhidden, dW, db = affine_backward(dscores, caches[self.num_layers-1])
    dW += self.reg * W
    grads['W' + str(self.num_layers)] = dW
    grads['b' + str(self.num_layers)] = db

    #decreasing for loop
    for i in range(self.num_layers-1,0,-1):
      if self.use_dropout:
        dhidden = dropout_backward(dhidden, dropout_caches[i-1])
      W = self.params['W' + str(i)]
      if self.use_batchnorm:
        dhidden, dW, db, dgamma, dbeta = affine_batchnorm_relu_backward(dhidden, caches[i-1])
        grads['gamma' + str(i)] = dgamma
        grads['beta' + str(i)] = dbeta
      else:
        dhidden, dW, db = affine_relu_backward(dhidden, caches[i-1])
      dW += self.reg * W
      grads['W' + str(i)] = dW
      grads['b' + str(i)] = db

    return loss, grads
