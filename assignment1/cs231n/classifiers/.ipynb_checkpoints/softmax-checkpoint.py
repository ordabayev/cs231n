import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    # scores[n,c] = sum[d]{X[n,d] * W[d,c]} 
    scores = X[i].dot(W)
    scores -= np.max(scores)
    # correct_class_score[n] = scores[n, y[n]]
    correct_class_score = scores[y[i]]
    # base[n] = sum[c]{exp(scores[n,c])}
    base = np.sum(np.exp(scores))
    # loss += sum[n]{-correct_class_score[n] + log(base[n])} / N
    loss += (-correct_class_score + np.log(base)) / num_train
    # (dloss/dloss) = 1
    dloss = 1
    # (dloss/dbase[n]) = (dloss/dloss) * (1/(base[n] * num_train))
    dbase = dloss/(base * num_train)
    # (dloss/dscores[n,c]) = (dloss/dbase[n]) * (exp(scores[n,c]))
    dscores = dbase*np.exp(scores)
    # (dloss/dscores[n, y[n]]) = (dloss/dloss) * (-1 / num_train)
    dscores[y[i]] -= dloss / num_train # from correct_class_score
    # (dloss/dW[d,c]) = sum[n]{(dloss/dscores[n,c]) * (X[n,d])}
    dW += np.dot(X[i][:,np.newaxis],dscores[np.newaxis,:])
  # loss += reg*sum[d,c]{W[d,c]*W[d,c]}
  loss += reg*np.sum(W*W)
  # (dloss/dW[n,c]) += (dloss/dloss) * (reg*2*W[n,c])
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
# scores[n,c] = sum[d]{X[n,d] * W[d,c]}
  scores = X.dot(W)
  scores -= np.amax(scores, axis=1, keepdims=True)
# correct_class_score[n] = scores[n, y[n]]
  correct_class_score = scores[np.arange(num_train), y]
# base[n] = sum[c]{exp(scores[n,c])}
  base = np.sum(np.exp(scores), axis=1)
# loss += mean[n]{-correct_class_score[n] + log(base[n])}
  loss += np.mean(-correct_class_score + np.log(base))
# loss += reg*sum[d,c]{W[d,c]*W[d,c]}
  loss += reg*np.sum(W*W)
  # Backpropagate gradient
# (dloss/dloss) = 1
  dloss = 1
# (dloss/dbase[n]) = (dloss/dloss) * (1/(base[n] * num_train))
  dbase = dloss/(base * num_train)
# (dloss/dscores[n,c]) = (dloss/dbase[n]) * (exp(scores[n,c]))
  dscores = dbase[:,np.newaxis]*np.exp(scores)
# (dloss/dscores[n, y[n]]) = (dloss/dloss) * (-1 / num_train)
  dscores[np.arange(num_train), y] -= dloss / num_train # correct_class_score
# (dloss/dW[d,c]) = sum[n]{(dloss/dscores[n,c]) * (X[n,d])}
  dW += np.dot(X.T,dscores)
# (dloss/dW[n,c]) += (dloss/dloss) * (reg*2*W[n,c])
  dW += 2*reg*W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

