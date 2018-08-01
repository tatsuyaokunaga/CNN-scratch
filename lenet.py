# coding: utf-8
import sys,os
import numpy as np

class LeNet:
    '''
    ネットワーク構成
    conv - relu - pool - conv - relu - pool
    affine - relu - affine - relu - affine - softmax 
    '''
    def __init__(self, input_dim,weight_init_std=0.01):
        
        #学習に必要なパラメータの設定
        W1 = weight_init_std * np.random.randn(6,1,5,5)
        b1 = np.zeros(6,)
        W2 = weight_init_std * np.random.randn(16,6,5,5)
        b2 = np.zeros(16)
        W3 = weight_init_std * np.random.randn(784, 120)
        b3 = np.zeros(120)
        W4 = weight_init_std * np.random.randn(120,84)
        b4 = np.zeros(84)
        W5 = weight_init_std* np.random.randn(84, 10)
        b5 = np.zeros(10)

        # レイヤの生成
        self.layers = [
                        Convolution(W1, b1,stride=1,pad=2),
                        Relu(),
                        Pooling(pool_h=2, pool_w=2, stride=2),
                        Convolution(W2, b2,stride=1, pad=2),
                        Relu(),
                        Pooling(pool_h=2, pool_w=2, stride=2),
                        Flatten(),
                        Affine(W3, b3),
                        Relu(),
                        Affine(W4,b4),
                        Relu(),
                        Affine(W5, b5)
                        ]
        
        self.loss_layer = SoftmaxWithLoss()
        
        self.params, self.grads = [], []
        for layer in (self.layers[0],self.layers[3],self.layers[7],self.layers[9],self.layers[11]):
            self.params += layer.params
            self.grads += layer.grads 
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def forward(self, x, t):
        score = self.predict(x) 
        loss = self.loss_layer.forward(score,t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)   
        return dout
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
