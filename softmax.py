import cPickle, gzip
import numpy as np

class SoftmaxRegression(object):
    '''
    A simple implementation of softmax regression using Numpy.
    The model is fit using gradient descent with momentum.

    Author: Matthew Green
    Date: 8/2/2016
    License: MIT
    '''
    def __init__(self, n_cat, intercept=True, one_hot=True, rate=0.1, p=0.5, max_iter=1000):
        self.options = { 'n_cat': n_cat, # number of categories
                         'intercept': intercept, # include intercept (bias)
                         'one_hot': one_hot, # transform y with one-hot encoding
                         'rate': rate, # learning rate
                         'p': p, # momentum coefficient
                         'max_iter': max_iter } # max. gradient descent steps

    def _one_hot(self, y):
        n_cat = self.options['n_cat']
        Yb = np.zeros((y.shape[0], n_cat))
        for i in xrange(y.shape[0]):
            Yb[i, y[i]] = 1.0
        return Yb

    def _add_intercept(self, X):
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    
    def fit(self, X, y):
        '''
        Inputs:
            X := (n samples, m features) matrix
            y := (n samples, ) matrix, with entries in range 0 to n_cat-1

        Notes:
            - X is transformed by adding an intercept column of ones
            if options['intercept'] is set.
            - y is transformed using one-hot encoding if options['one_hot']
            is set.
        '''
        options = self.options

        if options['intercept']:
            X_fit = self._add_intercept(X).T
        else:
            X_fit = X.T

        if options['one_hot']:
            y_fit = self._one_hot(y).T
        else:
            y_fit = y.T
        
        n_cat = options['n_cat']
        max_iter = options['max_iter']
        rate = options['rate']
        p = options['p']

        theta = np.zeros((n_cat, X_fit.shape[0]))
        dtheta = np.zeros(theta.shape)
        n_iter = 1
        while n_iter <= max_iter:
            z = np.dot(theta + p * dtheta, X_fit)
            zmax = np.amax(z, axis=0, keepdims=True)
            ez = np.exp(z - zmax)
            a = ez / np.sum(ez, axis=0, keepdims=True)
            grad = np.dot(a - y_fit, X_fit.T)
            dtheta = p * dtheta - rate * grad
            theta += dtheta
            n_iter += 1

        self.params = theta
        
    def predict(self, X):
        '''
        Input:
            X := (n samples, m features) matrix
        Output:
            yhat := (n samples, ) matrix, with entries in range 0 to n_cat-1
        '''
        if self.options['intercept']:
            X_pred = self._add_intercept(X).T
        else:
            X_pred = X.T
        z = np.dot(self.params, X_pred)
        zmax = np.amax(z, axis=0, keepdims=True)
        ez = np.exp(z - zmax)
        p = ez / np.sum(ez, axis=0, keepdims=True)
        return np.argmax(p, axis=0)

def test_mnist():
    ## Load local MNIST data set, downloaded from:
    ## http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = gzip.open('MNIST/mnist.pkl.gz', 'rb')
    train, valid, test = cPickle.load(f)
    f.close()

    ## Fit a softmax model with default options
    smr = SoftmaxRegression(n_cat=10)
    smr.fit(train[0], train[1])

    ## Evaluate test set error
    yhat = smr.predict(test[0])
    print np.mean(yhat == test[1]) # 0.9203

if __name__ == '__main__':
    test_mnist()
