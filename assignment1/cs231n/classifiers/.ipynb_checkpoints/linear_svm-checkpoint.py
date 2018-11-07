import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    dscores = np.zeros(scores.shape)
    correct_class_score = scores[y[i]]
    dcorrect_class_score = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dmargin = 1
      else:
        dmargin = 0
      dscores[j] = dmargin
      dcorrect_class_score -= dmargin
    dscores[y[i]] = dcorrect_class_score
    dW += np.dot(X[i][:,np.newaxis],dscores[np.newaxis,:])

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  # scores[n,c] = sum[d]{X[n,d]*W[d,c]} 
  scores = X.dot(W)
  # correct_class_score[n] = scores[n, y[n]]
  correct_class_score = scores[np.arange(num_train), y]
  # margin[n,c] = scores[n,c] - correct_class_score[n] + 1
  margin = scores - correct_class_score[:, np.newaxis] + 1 # note delta = 1
  # margin[n, y[n]] = 0
  margin[np.arange(num_train), y] = 0
  # loss = sum[n,c]{margin[n,c] > 0} / N
  loss = np.sum(margin[margin>0]) / num_train
  # loss += reg*sum[d,c]{W[d,c]*W[d,c]}
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # (dloss/dloss) = 1
  dloss = 1
  # (dloss/dmargin[n,c]) = (dloss/dloss) * (1/num_train) where margin[n,c] > 0
  dmargin = np.zeros_like(scores)
  dmargin[margin>0] = dloss / num_train
  # (dloss/dscores[n,c]) = (dloss/dmargin[n,c]) * (1)[n,c]
  dscores = dmargin
  # (dloss/dscores[n, y[n]]) = sum[c]{(dloss/dmargin[n,c]) * (-1)[n, y[n]]}
  dscores[np.arange(num_train), y] -= np.sum(dmargin, axis=1)
  # (dloss/dW[d,c]) += sum[n]{(dloss/dscores[n,c]) * (X[n,d])}
  dW += np.dot(X.T,dscores)
  # (dloss/dW[d,c]) += 2*reg*W[d,c]
  dW += 2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
