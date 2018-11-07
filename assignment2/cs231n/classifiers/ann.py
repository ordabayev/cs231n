import numpy as np
import tensorflow as tf
import math
import os
import inspect
import matplotlib.pyplot as plt

def reset_graph(seed=1):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

class NeuralNet(object):
    """ Neural Network
    """
    def __init__(self, network_architecture):
        self.NN = network_architecture
    
    def _create_graph(self):
        # input placeholders
        reset_graph()
        self.x = tf.placeholder(tf.float32, [None, *self.NN['n_input']], name='x')
        self.y = tf.placeholder(tf.int64, [None], name='y') # labels
        self.training = tf.placeholder_with_default(False, shape=(), name='training')
        # Create the classification network
        self.logits = self._classification_network()
        self.correct_prediction = tf.equal(tf.argmax(self.logits,1), self.y)
        self._create_loss_optimizer()
        # Initialize tensorflow variables
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
    
    def _classification_network(self):
        h_prev = self.x
        h = {}
        for layer in self.NN['layers']:
            if 'training' in inspect.signature(self.NN[layer]).parameters.keys():
                h[layer] = self.NN[layer](h_prev, training=self.training)
            else:
                h[layer] = self.NN[layer](h_prev)
            h_prev = h[layer]
        return h_prev
            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        #self.classif_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y,10), logits=self.logits)
        self.classif_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        self.loss = tf.reduce_mean(self.classif_loss, name='loss')   # average over batch
        # Use ADAM optimizer
        self.optimizer = self.NN['optimizer'](learning_rate=self.NN['learning_rate']).minimize(self.loss)
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    def _check_accuracy(self, x, y, sess, batch_size=100):
        n_samples = x.shape[0]
        num_batches = int(math.ceil(n_samples / batch_size))
        correct = 0
        for i in range(num_batches):
            x_batch = x[i*batch_size:(i+1)*batch_size]
            y_batch = y[i*batch_size:(i+1)*batch_size]
            corr = sess.run(self.correct_prediction, feed_dict={self.x: x_batch, self.y: y_batch})
            correct += np.sum(corr)
        accuracy = correct / n_samples
        return accuracy
    
    def train(self, x, y, x_val, y_val, restore=False, training_epochs=11, batch_size=100):
        self._create_graph()
        with tf.Session() as sess:
            if restore:
                print('Reloading existing')
                self.saver.restore(sess, './checkpoints/{}.ckpt'.format(self.NN['name']))
            else:
                print('Starting new')
                self.loss_history = []
                self.train_acc_history = []
                self.val_acc_history = []
                self.epochs = []
                self.init.run()
                
            restore_epochs = len(self.epochs)
            num_train = x.shape[0]
            for epoch in range(training_epochs):
                train_indices = np.arange(num_train)
                np.random.shuffle(train_indices)
                total_batch = int(math.ceil(num_train / batch_size))
                # Loop over all batches
                for i in range(total_batch):
                    idx = train_indices[batch_size*i:batch_size*(i+1)]
                    x_batch , y_batch = x[idx], y[idx]
                    # Fit training using batch data
                    opt, loss, _ = sess.run((self.optimizer, self.loss, self.extra_update_ops), 
                                  feed_dict={self.x: x_batch, self.y: y_batch, self.training: True})
                    self.loss_history.append(loss)
                    if i % 100 == 0:
                        print("\rIteration: {}/{}...".format(epoch*total_batch+i+1, total_batch*training_epochs), 
                              "Training loss: {:.4f}".format(loss), end="")
                # Calculate training and validation accuracies every epoch
                train_acc = self._check_accuracy(x, y, sess=sess)
                self.train_acc_history.append(train_acc)
                val_acc = self._check_accuracy(x_val, y_val, sess=sess)
                self.val_acc_history.append(val_acc)
                self.epochs.append(restore_epochs+epoch+1)
                
                print("\rEpoch: {}/{}...".format(restore_epochs+epoch+1, restore_epochs+training_epochs),
                      "train acc: {:.4f}; valid acc: {:.4f}".format(train_acc, val_acc))
            if os.path.isfile('./checkpoints/{}.ckpt.data-00000-of-00001'.format(self.NN['name'])):
                os.remove('./checkpoints/{}.ckpt.data-00000-of-00001'.format(self.NN['name']))
                os.remove('./checkpoints/{}.ckpt.index'.format(self.NN['name']))
                os.remove('./checkpoints/{}.ckpt.meta'.format(self.NN['name']))
            self.saver.save(sess, './checkpoints/{}.ckpt'.format(self.NN['name']))
            
    def plot(self):
        print('Name: {}'.format(self.NN['name']))
        print('Architecture: {}'.format(self.NN['layers']))
        print('Optimizer: {}'.format(self.NN['optimizer']))
        print('Learning rate: {}'.format(self.NN['learning_rate']))
        plt.figure(figsize=(15, 5))
        plt.subplot(1,2,1)
        plt.plot(self.loss_history)
        plt.xlabel('Iteration number')
        plt.ylabel('Loss')
        plt.subplot(1,2,2)
        plt.plot(self.epochs, self.train_acc_history, label='train acc')
        plt.plot(self.epochs,self.val_acc_history, label='valid acc')
        #plt.ylim(ymin=0)
        plt.legend()
        plt.xlabel('Epoch number')
        plt.ylabel('Accuracy')
        plt.show()