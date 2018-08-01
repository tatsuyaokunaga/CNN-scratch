# coding: utf-8
import sys,os
import numpy as np

class AlexNet:
    def __init__(self, input_dim,output_size =10):
        
        #学習に必要なパラメータの設定
        W1 = np.sqrt(2/96)*np.random.randn(96,3,11,11)
        b1 = np.sqrt(2/96)*np.zeros(96,)
        W2 =np.sqrt(2/256)* np.random.randn(256,96,5,5)
        b2 = np.sqrt(2/256)*np.zeros(256)
        W3 =np.sqrt(2/384)*np.random.randn(384,256,3,3)
        b3 = np.sqrt(2/384)*np.zeros(384)
        W4 =np.sqrt(2/384)*np.random.randn(384,384,3,3)
        b4 = np.sqrt(2/384)*np.zeros(384)
        W5 = np.sqrt(2/256)*np.random.randn(256,384,3,3)
        b5 = np.sqrt(2/256)* np.zeros(256)
        W6 = np.sqrt(2/2048)*np.random.randn(2304,2048)
        b6 = np.sqrt(2/2048)*np.zeros(2048)
        W7 = np.sqrt(2/2048)*np.random.randn(2048,2048)
        b7 = np.sqrt(2/2048)*np.zeros(2048)
        W8 = np.sqrt(2/10)*np.random.randn(2048,10)
        b8 = np.sqrt(2/10)* np.zeros(10)     

        # レイヤの生成
        self.layers = [
                                Convolution(W1, b1,stride=4,pad=52), #0
                                Relu(), #1
                                Pooling(pool_h=3, pool_w=3, stride=2), #2
                                BatchNormalization(gamma=1,beta=0), #3
                                Convolution(W2, b2,stride=1, pad=2), #4
                                Relu(), #5
                                Pooling(pool_h=3, pool_w=3, stride=2), #6
                                BatchNormalization(gamma=1,beta=0), #7
                                Convolution(W3, b3,stride=1, pad=1), #8
                                Relu(), #9
                                Convolution(W4, b4,stride=1, pad=1), #10
                                Relu(), #11
                                Convolution(W5, b5,stride=1, pad=1),#12
                                Relu(), #13
                                Pooling(pool_h=3, pool_w=3, stride=2), #14
                                BatchNormalization(gamma=1,beta=0), #15
                                Flatten(), #16
                                Affine(W6, b6), #17
                                Relu(), #18
                                Dropout(), #19
                                Affine(W7,b7), #20
                                Relu(), #21
                                Dropout(), #22
                                Affine(W8,b8) #23
                                ]
        
        self.loss_layer = SoftmaxWithLoss()
        
        self.params, self.grads = [], []
        for layer in (self.layers[0],self.layers[4],self.layers[8],self.layers[10],self.layers[12],self.layers[17],
                             self.layers[20],self.layers[23]):
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
         
