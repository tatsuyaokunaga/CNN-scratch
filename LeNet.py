# coding: utf-8
 
from layers import  Relu, Pooling, Convolution, Affine, SoftmaxWithLoss
from utils import *
from optimizer import *

import numpy as np
# from collections import OrderedDict

class LeNet:
    '''
    ネットワーク構成
    conv - relu - pool - conv - relu - pool
    affine - relu - affine - relu - affine - softmax 
    
    '''
    def __init__(self, input_dim =(3, 32, 32),
                           conv_param1={'filter_num':60, 'filter_size':5,'pad':2,'stride':1},
                           conv_param2={'filter_num':160, 'filter_size':5, 'pad':2, 'stride':1},
                           hidden_size1 = 120, hidden_size2 = 84, output_size =10,weight_init_std=0.01,
                           pool={'pool_h':2, 'pool_w':2, 'stride':2, 'pad':0}):
        

        input_size = input_dim[1]
        filter_num1 = conv_param1['filter_num']
        filter_size1 = conv_param1['filter_size']
        filter_pad1 = conv_param1['pad']
        filter_stride1 = conv_param1['stride']
        pool_w = pool['pool_w']
        pool_stride = pool['stride']
        conv_output_size1 = (input_size - filter_size1 + 2*filter_pad1) / filter_stride1 + 1
        pool_output1 = 1+int((conv_output_size1- pool_w)/pool_stride)
        filter_num2 = conv_param2['filter_num']
        filter_size2 = conv_param2['filter_size']
        filter_pad2 = conv_param2['pad']
        filter_stride2 = conv_param2['stride']
        input_size = pool_output1
        conv_output_size = (input_size - filter_size2 + 2*filter_pad2) / filter_stride2 + 1
        # 畳み込み層からのデータの出力サイズを計算
        pool_output_size = int(filter_num2 * (conv_output_size/2) * (conv_output_size/2))


        #学習に必要なパラメータの設定
        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param1, conv_param2]):
            self.params['W' + str(idx+1)] = weight_init_std * np.random.randn(conv_param['filter_num'], 
                                                                    pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
            
        self.params['W3'] = weight_init_std * np.random.randn(pool_output_size, hidden_size1)
        self.params['b3'] = np.zeros(hidden_size1)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size1,hidden_size2)
        self.params['b4'] = np.zeros(hidden_size2)
        self.params['W5'] = weight_init_std* np.random.randn(hidden_size2, output_size)
        self.params['b5'] = np.zeros(output_size)
        
        # レイヤの生成
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                               conv_param1['stride'], conv_param1['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                           conv_param2['stride'], conv_param2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Affine(self.params['W3'], self.params['b3']))
        self.layers.append(Relu())
        self.layers.append(Affine(self.params['W4'], self.params['b4']))
        self.layers.append(Relu())
        self.layers.append(Affine(self.params['W5'], self.params['b5']))
        
        self.last_layer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)   
        return x
        
    def loss(self, x, t):
        y  = self.predict(x) 
        return self.last_layer.forward(y,t)
    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers =self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i, layer_idx in enumerate((0, 3, 6, 8, 10)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db 

        return grads
