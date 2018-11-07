from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # hidden_layer[n,h] = sum[d]{X[n,d] * W1[d,h]} + b1[,h]
    hidden_layer = X.dot(W1) + b1[np.newaxis,:]
    # relu_layer[n,h] = max(0, hidden_layer[n,h])
    relu_layer = np.maximum(0, hidden_layer)
    # score[n,c] = sum[h]{relu_layer[n,h] * W2[h,c]} + b2[,c]
    scores = relu_layer.dot(W2) + b2[np.newaxis,:]
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    num_train = X.shape[0]
    scores -= np.amax(scores, axis=1, keepdims=True)
    # correct_class_score[n] = scores[n, y[n]]
    correct_class_score = scores[np.arange(num_train), y]
    # base[n] = sum[c]{exp(scores[n,c])}
    base = np.sum(np.exp(scores), axis=1)
    # loss = sum[n]{-correct_class_score[n] + log(base[n])} / N
    loss = np.sum(-correct_class_score + np.log(base)) / num_train
    # loss += reg*sum[d,c]{W[d,c]*W[d,c]}
    loss += reg*(np.sum(W1*W1) + np.sum(W2*W2))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # (dloss/dloss) = 1
    dloss = 1
    # (dloss/dbase[n]) = (dloss/dloss) * (1/(base[n] * num_train))
    dbase = dloss/(base * num_train)
    # (dloss/dscores[n,c]) = (dloss/dbase[n]) * (exp(scores[n,c]))
    dscores = dbase[:,np.newaxis]*np.exp(scores)
    # (dloss/dscores[n, y[n]]) = (dloss/dloss) * (-1 / num_train)
    dscores[np.arange(num_train), y] -= dloss / num_train # correct_class_score
    # (dloss/dW2[h,c]) = sum[n]{(dloss/dscores[n,c]) * (relu_layer[n,h])}
    dW2 = np.dot(relu_layer.T,dscores)
    # (dloss/db2[c]) = sum[n]{(dloss/dscores[n,c]) * 1[n]}
    db2 = np.sum(dscores, axis=0)
    # (dloss/dW2[h,c]) += (dloss/dloss) * (reg*2*W2[h,c])
    dW2 += 2*reg*W2
    
    # (dloss/drelu_layer[n,h]) = sum[c]{(dloss/dscores[n,c]) * (W2[h,c])}
    drelu_layer = dscores.dot(W2.T)
    # (dloss/dhidden_layer[n,h]) = (dloss/drelu_layer[n,h]) if hidden_layer > 0; else 0
    dhidden_layer = np.where(hidden_layer > 0, drelu_layer, 0)
    # (dloss/dW1[d,h]) = sum[n]{(dloss/dhidden_layer[n,h]) * (X[n,d])}
    dW1 = X.T.dot(dhidden_layer)
    # (dloss/db1[h]) = sum[n]{(dloss/dhidden_layer[n,h]) * (1[n])}
    db1 = np.sum(dhidden_layer, axis=0)
    # (dloss/dW1[d,h]) += (dloss/dloss) * (reg*2*W1[d,h])
    dW1 += 2*reg*W1
    
    grads['W1'], grads['b1'] = dW1, db1
    grads['W2'], grads['b2'] = dW2, db2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    self.loss_history = []
    self.train_acc_history = []
    self.val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      idx_batch = np.random.choice(num_train, batch_size)
      X_batch = X[idx_batch]
      y_batch = y[idx_batch]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      self.loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      for p in self.params:
        self.params[p] -= learning_rate*grads[p]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X) == y).mean() # train accuracy on entire X
        val_acc = (self.predict(X_val) == y_val).mean()
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': self.loss_history,
      'train_acc_history': self.train_acc_history,
      'val_acc_history': self.val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    scores = self.loss(X)
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred

  def history(self):
    # Plot the loss function and train / validation accuracies
    plt.figure(figsize=(15,4))
    plt.subplot(1, 2, 1)
    plt.plot(self.loss_history)
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.tight_layout() # tight layout

    plt.subplot(1, 2, 2)
    plt.plot(self.train_acc_history, label='train')
    plt.plot(self.val_acc_history, label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.tight_layout() # tight layout
    plt.show()


