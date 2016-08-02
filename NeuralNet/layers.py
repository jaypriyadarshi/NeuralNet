import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None

  N = x.shape[0]
  out = x.reshape(N, np.prod(x.shape[1:])).dot(w) + b

  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None

  N = x.shape[0]
  dx = np.dot(dout, w.T).reshape(x.shape)
  dw = np.dot(x.reshape(N, np.prod(x.shape[1:])).T, dout)
  db = np.sum(dout, axis=0)

  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None

  out = x * (x>0)

  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache

  dx = dout * (x>0)

  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':

    #break down all operations for easier backpropagation
    mean =  np.mean(x, axis=0)
    x_diff = x - mean
    x_diff_sq = x_diff**2
    variance = np.sum(x_diff_sq, axis=0)
    variance = variance / N
    sqrt_variance = np.sqrt(variance + eps)
    inverse_sqrt_var = 1.0 / sqrt_variance 
    #normalizing x => x_hat = (x - mean)/sqrt(variance + epsilon)
    x_hat = (x_diff) * inverse_sqrt_var
    #scale and shift
    scale_out = gamma * x_hat 
    out = scale_out + beta
    #compute running mean and variance
    running_mean = momentum * running_mean + (1.0 - momentum) * mean
    running_var = momentum * running_var + (1.0 - momentum) * variance
    cache = (mean, x_diff, x_diff_sq, variance, sqrt_variance, inverse_sqrt_var, x_hat, scale_out, gamma, beta, x, eps)

  elif mode == 'test':

    mean = running_mean
    variance = running_var
    x_hat = (x -  mean) / np.sqrt(variance + eps)
    out = gamma * x_hat + beta

  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  """
  Just to show the forward propagation steps

  mean =  np.mean(x, axis=0)
  x_diff = x - mean
  x_diff_sq = x_diff**2
  variance = np.sum(x_diff_sq, axis=0)
  variance = variance / N
  sqrt_variance = np.sqrt(variance + eps)
  inverse_sqrt_var = 1.0 / sqrt_variance 
  #normalizing x => x_hat = (x - mean)/sqrt(variance + epsilon)
  x_hat = (x_diff) * inverse_sqrt_var
  #scale and shift
  scale_out = gamma * x_hat 
  out = scale_out + beta
  """
  dx, dgamma, dbeta = None, None, None

  mean, x_diff, x_diff_sq, variance, sqrt_variance, inverse_sqrt_var, x_hat, scale_out, gamma, beta, x, eps = cache
  
  #backprop through all the operations in forward prop (shown in comments above) in reverse order
  N = x.shape[0]
  dscaled_out = dout
  dbeta = np.sum(dout, axis=0)
  dx_hat = gamma * dscaled_out
  dgamma = np.sum(x_hat * dscaled_out, axis=0)
  dx_diff = inverse_sqrt_var * dx_hat
  dinverse = np.sum(x_diff * dx_hat, axis=0)
  dsqrt_var = -1.0 / sqrt_variance**2
  dsqrt_var *= dinverse
  dvariance = 0.5 / np.sqrt(variance + eps)
  dvariance *= dsqrt_var
  dx_diff_sq = (1.0 / N) * np.ones((x_diff_sq.shape)) * dvariance
  dx_diff += 2 * x_diff * dx_diff_sq
  dx = dx_diff
  dmean = -np.sum(dx_diff, axis=0)
  dx += (1.0 / N) * np.ones((x.shape)) * dmean

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None

  mean, x_diff, x_diff_sq, variance, sqrt_variance, inverse_sqrt_var, x_hat, scale_out, gamma, beta, x, eps = cache
  N = x.shape[0]

  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(x_hat * dout, axis=0)
  #simplfying the gradient (As given in the paper) and taking the common terms out
  dx = (1.0 / N) * gamma * inverse_sqrt_var * (N * dout - np.sum(dout, axis=0) - (variance + eps)**(-1) * np.sum(dout * x_diff, axis=0) * x_diff)
   
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':

    mask = (np.random.rand(*x.shape) < p) / p
    out = x * mask

  elif mode == 'test':
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    dx = dout * mask
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None

  stride = conv_param['stride']
  pad = conv_param['pad']
  
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride

  padded_x = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')

  out = np.zeros((N, F, H_out, W_out))

  for i in range(N):
  	for j in range(F):
  		for k in range(H_out):
  			for l in range(W_out):
  				out[i, j, k, l] = np.sum(w[j, :] * padded_x[i, :, stride * k:stride * k + HH, stride * l:stride * l + WW]) + b[j]


  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  x, w, b, conv_param = cache

  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  stride = conv_param['stride']
  pad = conv_param['pad']

  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride

  padded_x = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  padded_dx = np.pad(dx, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')

  for i in range(N):
  	for j in range(F):
  		for k in range(H_out):
  			for l in range(W_out):
  				padded_dx[i, :, stride * k:stride * k + HH, stride * l:stride * l + WW] += w[j] * dout[i, j, k, l]
  				#extract receptive field and take product with dout
  				dw[j] += padded_x[i, :, stride * k:stride * k + HH, stride * l:stride * l + WW] * dout[i, j, k, l]
  				db[j] += dout[i, j, k, l]

  #as we padded dx earlier, we need to consider the sub-matrix belonging to actual x's dimensions
  dx = padded_dx[:, :, pad:pad+H, pad:pad+W]			

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None

  N, C, H, W = x.shape

  stride = pool_param['stride']
  PH = pool_param['pool_height']
  PW = pool_param['pool_width']

  H_out = 1 + (H - PH) / stride
  W_out = 1 + (W - PW) / stride

  out = np.zeros((N, C, H_out, W_out))

  for i in range(N):
  	for j in range(C):
  		for k in range(H_out):
  			for l in range(W_out):
  				out[i, j, k, l] = np.max(x[i, j, stride * k:stride * k + PH, stride * l:stride * l + PW])

  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None

  x, pool_param = cache

  N, C, H, W = x.shape

  PH = pool_param['pool_height']
  PW = pool_param['pool_width']
  stride = pool_param['stride']

  H_out = 1 + (H - PH) / stride
  W_out = 1 + (W - PW) / stride

  dx = np.zeros_like(x)

  for i in range(N):
  	for j in range(C):
  		for k in range(H_out):
  			for l in range(W_out):
  				# result from max pool layer, the max element's derivate will be 1 rest will be 0
  				receptive_field = x[i, j, stride * k:stride * k + PH, stride * l:stride * l + PW]
  				pool_result = np.max(receptive_field)
  				# Step to only consider contribution from the max element of the receptive field
  				dx[i, j, stride * k:stride * k + PH, stride * l:stride * l + PW] += (receptive_field == pool_result) * dout[i, j, k, l]

  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, C, H, W = x.shape

  running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

  if mode == 'train':
  	mean =  (1.0 / (N * H * W) * np.sum(x, axis=(0,2,3))).reshape(1, C, 1, 1)
  	variance = (1.0 / (N * H * W) * np.sum((x-mean)**2, axis=(0,2,3))).reshape(1, C, 1, 1)
  	#normalizing x => x_hat = (x - mean)/sqrt(variance + epsilon)
  	x_hat = (x - mean) / np.sqrt(variance + eps)
  	#scale and shift 
  	out = gamma.reshape(1, C, 1, 1) * x_hat + beta.reshape(1, C, 1, 1)
  	#compute running mean and variance
  	running_mean = momentum * running_mean + (1.0 - momentum) * np.squeeze(mean)
  	running_var = momentum * running_var + (1.0 - momentum) * np.squeeze(variance)
  	cache = (mean, variance, x_hat, gamma, beta, x, eps)

  	bn_param['running_mean'] = running_mean
  	bn_param['running_var'] = running_var

  elif mode == 'test':
  	mean = running_mean.reshape(1, C, 1, 1)
  	variance = running_var.reshape(1, C, 1, 1)
  	x_hat = (x - mean) / np.sqrt(variance + eps)
  	out = gamma.reshape(1, C, 1, 1) * x_hat + beta.reshape(1, C, 1, 1)
  
  else:
  	raise ValueError('Invalid forward batchnorm mode "%s"' % mode)


  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  mean, variance, x_hat, gamma, beta, x, eps = cache
  N, C, H, W = x.shape

  dbeta = np.sum(dout, axis=(0, 2, 3))
  dgamma = np.sum(x_hat * dout, axis=(0, 2, 3))
  inverse_sqrt_var = 1.0 / np.sqrt(variance + eps)
  x_diff = x - mean
  #simplfying the gradient (As given in the paper) and taking the common terms out
  dx = (1.0 / (N * H * W)) * gamma.reshape(1, C, 1, 1) * inverse_sqrt_var * ((N * H * W) * dout - np.sum(dout, axis=(0, 2, 3)).reshape(1, C, 1, 1) - (variance + eps)**(-1) * np.sum(dout * x_diff, axis=(0, 2,3)).reshape(1, C, 1, 1) * x_diff)

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
