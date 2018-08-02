{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CNNの説明"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "DNNと比較すると、「畳み込み層（Convolutionレイヤ）」と「プーリング層（Poolingレイヤ）」が新たに加わる。  \n",
    "また、出力に近い層ではDNNと同様に全結合層が用いられる。一方、入力層では、「畳み込み層」+「ReLU」+「プーリング層」などの流れになることが大きな違い。  \n",
    "「畳み込み層」では入出力データに当たる「特徴マップ」に、重みに対応するフィルターを適用して「畳み込み演算」を施す。  \n",
    "その際に、通常、パディング（padding）という処理を施し、元の配列のまわりをある値で埋めてから演算を施す。  \n",
    "また、プーリング層とは、特徴マップの情報を圧縮する役割を持つ層こと。\n",
    "畳み込み層での演算により、画像データの三次元データの形状を維持できるため、画像認識や音声認識など至る所で使われている。 "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ChainerによるLeNetの実装"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import chainer\n",
    "from chainer.backends import cuda\n",
    "from chainer import Function, gradient_check, report, training, utils, Variable\n",
    "from chainer import datasets, iterators, optimizers, serializers\n",
    "from chainer import Link, Chain, ChainList\n",
    "import chainer.functions as F\n",
    "import chainer.links as L\n",
    "from chainer.training import extensions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "import argparse\n",
    "from chainer import Link, Chain, ChainList\n",
    "import chainer\n",
    "import chainer.functions as F\n",
    "import chainer.links as L\n",
    "from chainer import training\n",
    "from chainer.training import extensions\n",
    "\n",
    "# Network definition\n",
    "class LeNet(chainer.Chain):\n",
    "\n",
    "    def __init__(self, class_num=10):\n",
    "        super(LeNet, self).__init__()\n",
    "        with self.init_scope():\n",
    "            # the size of the inputs to each layer will be inferred\n",
    "            self.conv1 = L.Convolution2D(None, 6, 5, stride=1)\n",
    "            self.conv2 = L.Convolution2D(None, 16, 5,stride=1)\n",
    "            self.fc3 = L.Linear(None, 120)\n",
    "            self.fc4 = L.Linear(None, 84)\n",
    "            self.fc5 = L.Linear(None, class_num)\n",
    "\n",
    "    def __call__(self, x):\n",
    "        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)\n",
    "        h = F.max_pooling_2d(F.relu(self.conv2(pool1)), ksize=2, stride=2)\n",
    "        h = F.relu(self.fc3(h))\n",
    "        h = F.relu(self.fc4(h))    \n",
    "        h = self.fc5(h) \n",
    "\n",
    "        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "# スクラッチで構築したCNN(LeNet)ネットワークで学習（データセットはmnist）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import sys\n",
    "sys.path.append('..')\n",
    "import numpy as np\n",
    "import keras\n",
    "from keras.datasets import mnist\n",
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib import cm\n",
    "%matplotlib inline\n",
    "import random\n",
    "\n",
    "max_epoch =50\n",
    "batch_size =16\n",
    "learning_rate = 0.001\n",
    "\n",
    "(x_train, t_train), (x_test, t_test) = mnist.load_data()\n",
    "\n",
    "x_train = x_train.reshape(60000,1,28,28) # 2次元配列を1次元に変換\n",
    "x_test = x_test.reshape(10000, 1,28,28)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 1, 28, 28)"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 0〜1の範囲に正規化\n",
    "x_train = x_train.astype('float32')\n",
    "x_test = x_test.astype('float32')\n",
    "# x_train /= 255\n",
    "# x_test /= 255\n",
    "# データ数1/10で学習\n",
    "x_train, t_train = x_train[:6000], t_train[:6000]\n",
    "x_test, t_test = x_test[:1000], t_test[:1000]\n",
    "num_classes = 10\n",
    "t_train = keras.utils.to_categorical(t_train, num_classes)\n",
    "t_test = keras.utils.to_categorical(t_test, num_classes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "| epoch 1 | iter 375 / 375 | loss 2.3025 | train_acc 0.1543 | test_acc 0.1730\n",
      "| epoch 2 | iter 375 / 375 | loss 2.3023 | train_acc 0.1118 | test_acc 0.1260\n",
      "| epoch 3 | iter 375 / 375 | loss 2.3021 | train_acc 0.1118 | test_acc 0.1260\n",
      "| epoch 4 | iter 375 / 375 | loss 2.3019 | train_acc 0.1118 | test_acc 0.1260\n",
      "| epoch 5 | iter 375 / 375 | loss 2.3016 | train_acc 0.1118 | test_acc 0.1260\n",
      "| epoch 6 | iter 375 / 375 | loss 2.3014 | train_acc 0.1118 | test_acc 0.1260\n",
      "| epoch 7 | iter 375 / 375 | loss 2.3012 | train_acc 0.1118 | test_acc 0.1260\n",
      "| epoch 8 | iter 375 / 375 | loss 2.3008 | train_acc 0.1118 | test_acc 0.1260\n",
      "| epoch 9 | iter 375 / 375 | loss 2.3004 | train_acc 0.1162 | test_acc 0.1270\n",
      "| epoch 10 | iter 375 / 375 | loss 2.2997 | train_acc 0.1582 | test_acc 0.1560\n",
      "| epoch 11 | iter 375 / 375 | loss 2.2983 | train_acc 0.2090 | test_acc 0.2110\n",
      "| epoch 12 | iter 375 / 375 | loss 2.2946 | train_acc 0.3365 | test_acc 0.3310\n",
      "| epoch 13 | iter 375 / 375 | loss 2.2773 | train_acc 0.1712 | test_acc 0.1710\n",
      "| epoch 14 | iter 375 / 375 | loss 1.9934 | train_acc 0.5115 | test_acc 0.5220\n",
      "| epoch 15 | iter 375 / 375 | loss 0.9418 | train_acc 0.6822 | test_acc 0.6650\n",
      "| epoch 16 | iter 375 / 375 | loss 0.6395 | train_acc 0.8297 | test_acc 0.8150\n",
      "| epoch 17 | iter 375 / 375 | loss 0.4926 | train_acc 0.8580 | test_acc 0.8330\n",
      "| epoch 18 | iter 375 / 375 | loss 0.3915 | train_acc 0.8945 | test_acc 0.8790\n",
      "| epoch 19 | iter 375 / 375 | loss 0.3346 | train_acc 0.9218 | test_acc 0.8990\n",
      "| epoch 20 | iter 375 / 375 | loss 0.2801 | train_acc 0.9282 | test_acc 0.9130\n",
      "| epoch 21 | iter 375 / 375 | loss 0.2462 | train_acc 0.9425 | test_acc 0.9260\n",
      "| epoch 22 | iter 375 / 375 | loss 0.2128 | train_acc 0.9438 | test_acc 0.9150\n",
      "| epoch 23 | iter 375 / 375 | loss 0.1812 | train_acc 0.9495 | test_acc 0.9240\n",
      "| epoch 24 | iter 375 / 375 | loss 0.1623 | train_acc 0.9628 | test_acc 0.9460\n",
      "| epoch 25 | iter 375 / 375 | loss 0.1455 | train_acc 0.9603 | test_acc 0.9410\n",
      "| epoch 26 | iter 375 / 375 | loss 0.1303 | train_acc 0.9612 | test_acc 0.9310\n",
      "| epoch 27 | iter 375 / 375 | loss 0.1178 | train_acc 0.9683 | test_acc 0.9460\n",
      "| epoch 28 | iter 375 / 375 | loss 0.1046 | train_acc 0.9713 | test_acc 0.9500\n",
      "| epoch 29 | iter 375 / 375 | loss 0.0937 | train_acc 0.9743 | test_acc 0.9520\n",
      "| epoch 30 | iter 375 / 375 | loss 0.0906 | train_acc 0.9807 | test_acc 0.9650\n",
      "| epoch 31 | iter 375 / 375 | loss 0.0806 | train_acc 0.9845 | test_acc 0.9580\n",
      "| epoch 32 | iter 375 / 375 | loss 0.0722 | train_acc 0.9812 | test_acc 0.9590\n",
      "| epoch 33 | iter 375 / 375 | loss 0.0617 | train_acc 0.9808 | test_acc 0.9540\n",
      "| epoch 34 | iter 375 / 375 | loss 0.0616 | train_acc 0.9852 | test_acc 0.9650\n",
      "| epoch 35 | iter 375 / 375 | loss 0.0582 | train_acc 0.9870 | test_acc 0.9640\n",
      "| epoch 36 | iter 375 / 375 | loss 0.0549 | train_acc 0.9880 | test_acc 0.9660\n",
      "| epoch 37 | iter 375 / 375 | loss 0.0467 | train_acc 0.9887 | test_acc 0.9660\n",
      "| epoch 38 | iter 375 / 375 | loss 0.0452 | train_acc 0.9903 | test_acc 0.9660\n",
      "| epoch 39 | iter 375 / 375 | loss 0.0355 | train_acc 0.9910 | test_acc 0.9650\n",
      "| epoch 40 | iter 375 / 375 | loss 0.0374 | train_acc 0.9897 | test_acc 0.9580\n",
      "| epoch 41 | iter 375 / 375 | loss 0.0324 | train_acc 0.9813 | test_acc 0.9420\n",
      "| epoch 42 | iter 375 / 375 | loss 0.0311 | train_acc 0.9950 | test_acc 0.9610\n",
      "| epoch 43 | iter 375 / 375 | loss 0.0291 | train_acc 0.9942 | test_acc 0.9650\n",
      "| epoch 44 | iter 375 / 375 | loss 0.0237 | train_acc 0.9922 | test_acc 0.9620\n",
      "| epoch 45 | iter 375 / 375 | loss 0.0232 | train_acc 0.9957 | test_acc 0.9690\n",
      "| epoch 46 | iter 375 / 375 | loss 0.0222 | train_acc 0.9962 | test_acc 0.9690\n",
      "| epoch 47 | iter 375 / 375 | loss 0.0184 | train_acc 0.9965 | test_acc 0.9680\n",
      "| epoch 48 | iter 375 / 375 | loss 0.0158 | train_acc 0.9953 | test_acc 0.9670\n",
      "| epoch 49 | iter 375 / 375 | loss 0.0145 | train_acc 0.9923 | test_acc 0.9600\n",
      "| epoch 50 | iter 375 / 375 | loss 0.0146 | train_acc 0.9970 | test_acc 0.9580\n",
      "max_test_acc : 0.969\n"
     ]
    }
   ],
   "source": [
    "from lenet import lenet\n",
    "from layers import *\n",
    "from optimizer import *\n",
    "from utils import *\n",
    "# モデルとoptimizerの生成\n",
    "model = LeNet(input_dim =(1,28,28))\n",
    "optimizer = SGD(lr=learning_rate)\n",
    "\n",
    "data_size = len(x_train)\n",
    "max_iter = data_size // batch_size\n",
    "total_loss = 0\n",
    "count =0\n",
    "loss_list =[]\n",
    "train_acc_list = []\n",
    "test_acc_list = []\n",
    "\n",
    "\n",
    "for epoch in range(max_epoch):\n",
    "    idx = np.random.permutation(data_size)\n",
    "    x_train = x_train[idx]\n",
    "    t_train = t_train[idx]\n",
    "    \n",
    "    for iters in range(max_iter):\n",
    "        xx =x_train[iters*batch_size:(iters+1)*batch_size]\n",
    "        tt = t_train[iters*batch_size:(iters+1)*batch_size]\n",
    "       \n",
    "        # 勾配を求め、パラメータを更新\n",
    "        loss = model.forward(xx,tt)\n",
    "        model.backward()\n",
    "        optimizer.update(model.params,model.grads)\n",
    "\n",
    "        total_loss +=loss\n",
    "        count +=1\n",
    "        model.accuracy(xx,tt)\n",
    "    \n",
    "        if (iters +1)%max_iter ==0:\n",
    "            avg_loss = total_loss/count\n",
    "            train_acc = model.accuracy(x_train, t_train)\n",
    "            test_acc = model.accuracy(x_test, t_test)\n",
    "            print('| epoch %d | iter %d / %d | loss %.4f | train_acc %.4f | test_acc %.4f' \n",
    "                      %(epoch + 1, iters + 1, max_iter, avg_loss, train_acc,test_acc))\n",
    "            loss_list.append(avg_loss)\n",
    "            train_acc_list.append(train_acc)\n",
    "            test_acc_list.append(test_acc)\n",
    "            total_loss, count = 0, 0\n",
    "print('max_test_acc :',max(test_acc_list))           "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " ## データ数を1/10に絞り学習した結果、検証データのaccuracyが約97%\n",
    " "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# mnistをAlexNet (keras)で学習"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 296,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "import numpy as np\n",
    "import keras\n",
    "from keras.datasets import mnist\n",
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib import cm\n",
    "%matplotlib inline\n",
    "import random\n",
    "\n",
    "learning_rate = 0.001\n",
    "\n",
    "(x_train, t_train), (x_test, t_test) = mnist.load_data()\n",
    "\n",
    "x_train = x_train.reshape(60000,1,28,28) # 2次元配列を1次元に変換\n",
    "x_test = x_test.reshape(10000, 1,28,28)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 297,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 0〜1の範囲に正規化\n",
    "x_train = x_train.astype('float32')\n",
    "x_test = x_test.astype('float32')\n",
    "# x_train /= 255\n",
    "# x_test /= 255\n",
    "# データ数1/10で学習\n",
    "x_train, t_train = x_train[:6000], t_train[:6000]\n",
    "x_test, t_test = x_test[:1000], t_test[:1000]\n",
    "num_classes = 10\n",
    "t_train = keras.utils.to_categorical(t_train, num_classes)\n",
    "t_test = keras.utils.to_categorical(t_test, num_classes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 299,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import  Sequential\n",
    "\n",
    "from keras.layers import *\n",
    "\n",
    "model = Sequential()\n",
    "\n",
    "model.add(Conv2D(96, kernel_size=(11, 11),\n",
    "                 strides=(4, 4),\n",
    "                 padding='same',\n",
    "                 activation='relu',\n",
    "                 input_shape=(1,28,28)))\n",
    "model.add(MaxPooling2D(pool_size=(3, 3),\n",
    "                      strides=(2,2),\n",
    "                      padding='same'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(Conv2D(256, kernel_size=(5, 5),\n",
    "                 strides=(1, 1),\n",
    "                 padding='same',\n",
    "                 activation='relu'))\n",
    "model.add(MaxPooling2D(pool_size=(3, 3),\n",
    "                      strides=(2,2),\n",
    "                      padding='same'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(Conv2D(384, kernel_size=(3, 3),\n",
    "                 strides=(1, 1),\n",
    "                 padding='same',\n",
    "                 activation='relu'))\n",
    "model.add(Conv2D(384, kernel_size=(3, 3),\n",
    "                 strides=(1, 1),\n",
    "                 padding='same',\n",
    "                 activation='relu'))\n",
    "model.add(Conv2D(256, kernel_size=(3, 3),\n",
    "                 strides=(1, 1),\n",
    "                 padding='same',\n",
    "                 activation='relu'))\n",
    "model.add(MaxPooling2D(pool_size=(3, 3),\n",
    "                      strides=(2,2),\n",
    "                      padding='same'))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(2048, activation='relu'))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(2048, activation='relu'))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(num_classes, activation='softmax'))\n",
    "\n",
    "model.compile(loss=keras.losses.categorical_crossentropy,\n",
    "              optimizer=keras.optimizers.SGD(),\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 300,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "6000/6000 [==============================] - 55s 9ms/step - loss: 0.6766 - acc: 0.7962\n",
      "Epoch 2/50\n",
      "6000/6000 [==============================] - 53s 9ms/step - loss: 0.2529 - acc: 0.9245\n",
      "Epoch 3/50\n",
      "6000/6000 [==============================] - 54s 9ms/step - loss: 0.1569 - acc: 0.9538\n",
      "Epoch 4/50\n",
      "6000/6000 [==============================] - 54s 9ms/step - loss: 0.1147 - acc: 0.9658\n",
      "Epoch 5/50\n",
      "6000/6000 [==============================] - 53s 9ms/step - loss: 0.1088 - acc: 0.9660\n",
      "Epoch 6/50\n",
      "6000/6000 [==============================] - 54s 9ms/step - loss: 0.0675 - acc: 0.9788\n",
      "Epoch 7/50\n",
      "6000/6000 [==============================] - 53s 9ms/step - loss: 0.0740 - acc: 0.9775\n",
      "Epoch 8/50\n",
      "6000/6000 [==============================] - 56s 9ms/step - loss: 0.0609 - acc: 0.9787\n",
      "Epoch 9/50\n",
      "6000/6000 [==============================] - 58s 10ms/step - loss: 0.0559 - acc: 0.9820\n",
      "Epoch 10/50\n",
      "6000/6000 [==============================] - 58s 10ms/step - loss: 0.0382 - acc: 0.9885\n",
      "Epoch 11/50\n",
      "6000/6000 [==============================] - 56s 9ms/step - loss: 0.0515 - acc: 0.9823\n",
      "Epoch 12/50\n",
      "6000/6000 [==============================] - 55s 9ms/step - loss: 0.0363 - acc: 0.9898\n",
      "Epoch 13/50\n",
      "6000/6000 [==============================] - 56s 9ms/step - loss: 0.0217 - acc: 0.9933\n",
      "Epoch 14/50\n",
      "6000/6000 [==============================] - 56s 9ms/step - loss: 0.0239 - acc: 0.9917\n",
      "Epoch 15/50\n",
      "6000/6000 [==============================] - 58s 10ms/step - loss: 0.0234 - acc: 0.9935\n",
      "Epoch 16/50\n",
      "6000/6000 [==============================] - 57s 9ms/step - loss: 0.0171 - acc: 0.9952\n",
      "Epoch 17/50\n",
      "6000/6000 [==============================] - 57s 10ms/step - loss: 0.0167 - acc: 0.9948\n",
      "Epoch 18/50\n",
      "6000/6000 [==============================] - 57s 9ms/step - loss: 0.0144 - acc: 0.9957\n",
      "Epoch 19/50\n",
      "6000/6000 [==============================] - 57s 10ms/step - loss: 0.0133 - acc: 0.9958\n",
      "Epoch 20/50\n",
      "6000/6000 [==============================] - 56s 9ms/step - loss: 0.0121 - acc: 0.9968\n",
      "Epoch 21/50\n",
      "6000/6000 [==============================] - 55s 9ms/step - loss: 0.0157 - acc: 0.9955\n",
      "Epoch 22/50\n",
      "6000/6000 [==============================] - 56s 9ms/step - loss: 0.0093 - acc: 0.9970\n",
      "Epoch 23/50\n",
      "6000/6000 [==============================] - 55s 9ms/step - loss: 0.0149 - acc: 0.9955\n",
      "Epoch 24/50\n",
      "6000/6000 [==============================] - 55s 9ms/step - loss: 0.0073 - acc: 0.9980\n",
      "Epoch 25/50\n",
      "6000/6000 [==============================] - 57s 10ms/step - loss: 0.0083 - acc: 0.9977\n",
      "Epoch 26/50\n",
      "6000/6000 [==============================] - 59s 10ms/step - loss: 0.0114 - acc: 0.9970\n",
      "Epoch 27/50\n",
      "6000/6000 [==============================] - 58s 10ms/step - loss: 0.0081 - acc: 0.9970\n",
      "Epoch 28/50\n",
      "6000/6000 [==============================] - 57s 10ms/step - loss: 0.0099 - acc: 0.9970\n",
      "Epoch 29/50\n",
      "6000/6000 [==============================] - 56s 9ms/step - loss: 0.0126 - acc: 0.9960\n",
      "Epoch 30/50\n",
      "6000/6000 [==============================] - 55s 9ms/step - loss: 0.0081 - acc: 0.9975\n",
      "Epoch 31/50\n",
      "6000/6000 [==============================] - 57s 9ms/step - loss: 0.0072 - acc: 0.9980\n",
      "Epoch 32/50\n",
      "6000/6000 [==============================] - 56s 9ms/step - loss: 0.0034 - acc: 0.9992\n",
      "Epoch 33/50\n",
      "6000/6000 [==============================] - 58s 10ms/step - loss: 0.0030 - acc: 0.9992\n",
      "Epoch 34/50\n",
      "6000/6000 [==============================] - 57s 10ms/step - loss: 0.0041 - acc: 0.9990\n",
      "Epoch 35/50\n",
      "6000/6000 [==============================] - 57s 9ms/step - loss: 0.0070 - acc: 0.9983\n",
      "Epoch 36/50\n",
      "6000/6000 [==============================] - 56s 9ms/step - loss: 0.0091 - acc: 0.9972\n",
      "Epoch 37/50\n",
      "6000/6000 [==============================] - 57s 10ms/step - loss: 0.0048 - acc: 0.9985\n",
      "Epoch 38/50\n",
      "6000/6000 [==============================] - 57s 9ms/step - loss: 0.0092 - acc: 0.9977\n",
      "Epoch 39/50\n",
      "6000/6000 [==============================] - 56s 9ms/step - loss: 0.0048 - acc: 0.9987\n",
      "Epoch 40/50\n",
      "6000/6000 [==============================] - 55s 9ms/step - loss: 0.0058 - acc: 0.9982\n",
      "Epoch 41/50\n",
      "6000/6000 [==============================] - 58s 10ms/step - loss: 0.0085 - acc: 0.9975\n",
      "Epoch 42/50\n",
      "6000/6000 [==============================] - 58s 10ms/step - loss: 0.0057 - acc: 0.9980\n",
      "Epoch 43/50\n",
      "6000/6000 [==============================] - 56s 9ms/step - loss: 0.0057 - acc: 0.9983\n",
      "Epoch 44/50\n",
      "6000/6000 [==============================] - 56s 9ms/step - loss: 0.0021 - acc: 0.9995\n",
      "Epoch 45/50\n",
      "6000/6000 [==============================] - 56s 9ms/step - loss: 0.0032 - acc: 0.9987\n",
      "Epoch 46/50\n",
      "6000/6000 [==============================] - 57s 9ms/step - loss: 0.0041 - acc: 0.9983\n",
      "Epoch 47/50\n",
      "6000/6000 [==============================] - 58s 10ms/step - loss: 0.0051 - acc: 0.9983\n",
      "Epoch 48/50\n",
      "6000/6000 [==============================] - 55s 9ms/step - loss: 0.0077 - acc: 0.9977\n",
      "Epoch 49/50\n",
      "6000/6000 [==============================] - 54s 9ms/step - loss: 0.0079 - acc: 0.9982\n",
      "Epoch 50/50\n",
      "6000/6000 [==============================] - 55s 9ms/step - loss: 0.0116 - acc: 0.9970\n",
      "1000/1000 [==============================] - 1s 817us/step\n",
      "1000/1000 [==============================] - 0s 499us/step\n",
      "acc: 95.40%\n"
     ]
    }
   ],
   "source": [
    "model.fit(x_train, t_train,\n",
    "          epochs=50,\n",
    "          batch_size=16)\n",
    "score = model.evaluate(x_test, t_test,batch_size=32)\n",
    "print(\"%s: %.2f%%\" % (model.metrics_names[1], score[1] * 100))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## testデータで検証した結果、accuracyは95.4%"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# スクラッチで構築したAlexNetで学習（データセットはcifar10）"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ミニバッチ学習"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 357,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "import numpy as np\n",
    "import keras\n",
    "from keras.datasets import cifar10\n",
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib import cm\n",
    "%matplotlib inline\n",
    "import random\n",
    "\n",
    "max_epoch = 50\n",
    "batch_size =32\n",
    "learning_rate = 0.001\n",
    "\n",
    "(x_train, t_train), (x_test, t_test) = cifar10.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 358,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((50000, 32, 32, 3), (10000, 32, 32, 3))"
      ]
     },
     "execution_count": 358,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape,x_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 359,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 0〜1の範囲に正規化\n",
    "x_train = x_train.transpose(0,3,1,2)\n",
    "x_test = x_test.transpose(0,3,1,2)\n",
    "x_train = x_train.astype('float32')\n",
    "x_test = x_test.astype('float32')\n",
    "x_train /= 255\n",
    "x_test /= 255\n",
    "# データ数1/10で学習\n",
    "x_train, t_train = x_train[:1000], t_train[:1000]\n",
    "x_test, t_test = x_test[:200], t_test[:200]\n",
    "\n",
    "num_classes = 10\n",
    "t_train = keras.utils.to_categorical(t_train, num_classes)\n",
    "t_test = keras.utils.to_categorical(t_test, num_classes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 360,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1000, 3, 32, 32)"
      ]
     },
     "execution_count": 360,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 361,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "| epoch 1 | iter 31 / 31 | loss 10.4867 | train_acc 0.2230 | test_acc 0.1750\n",
      "| epoch 2 | iter 31 / 31 | loss 6.8676 | train_acc 0.2260 | test_acc 0.1700\n",
      "| epoch 3 | iter 31 / 31 | loss 3.5849 | train_acc 0.2500 | test_acc 0.1350\n",
      "| epoch 4 | iter 31 / 31 | loss 2.5800 | train_acc 0.2500 | test_acc 0.1600\n",
      "| epoch 5 | iter 31 / 31 | loss 2.2237 | train_acc 0.2980 | test_acc 0.2100\n",
      "| epoch 6 | iter 31 / 31 | loss 2.0598 | train_acc 0.3410 | test_acc 0.2400\n",
      "| epoch 7 | iter 31 / 31 | loss 1.9365 | train_acc 0.3480 | test_acc 0.2350\n",
      "| epoch 8 | iter 31 / 31 | loss 1.8783 | train_acc 0.3670 | test_acc 0.2550\n",
      "| epoch 9 | iter 31 / 31 | loss 1.8627 | train_acc 0.3880 | test_acc 0.2650\n",
      "| epoch 10 | iter 31 / 31 | loss 1.8268 | train_acc 0.3800 | test_acc 0.2500\n",
      "| epoch 11 | iter 31 / 31 | loss 1.7637 | train_acc 0.4000 | test_acc 0.2400\n",
      "| epoch 12 | iter 31 / 31 | loss 1.7817 | train_acc 0.4430 | test_acc 0.2300\n",
      "| epoch 13 | iter 31 / 31 | loss 1.7273 | train_acc 0.4430 | test_acc 0.3050\n",
      "| epoch 14 | iter 31 / 31 | loss 1.7251 | train_acc 0.4470 | test_acc 0.2350\n",
      "| epoch 15 | iter 31 / 31 | loss 1.7143 | train_acc 0.4600 | test_acc 0.2800\n",
      "| epoch 16 | iter 31 / 31 | loss 1.7264 | train_acc 0.4460 | test_acc 0.3050\n",
      "| epoch 17 | iter 31 / 31 | loss 1.6606 | train_acc 0.4610 | test_acc 0.2550\n",
      "| epoch 18 | iter 31 / 31 | loss 1.6372 | train_acc 0.4820 | test_acc 0.2500\n",
      "| epoch 19 | iter 31 / 31 | loss 1.5764 | train_acc 0.5270 | test_acc 0.2600\n",
      "| epoch 20 | iter 31 / 31 | loss 1.4378 | train_acc 0.5430 | test_acc 0.2750\n",
      "| epoch 21 | iter 31 / 31 | loss 1.4166 | train_acc 0.5970 | test_acc 0.2700\n",
      "| epoch 22 | iter 31 / 31 | loss 1.2895 | train_acc 0.6110 | test_acc 0.2900\n",
      "| epoch 23 | iter 31 / 31 | loss 1.2805 | train_acc 0.6200 | test_acc 0.2450\n",
      "| epoch 24 | iter 31 / 31 | loss 1.2490 | train_acc 0.6410 | test_acc 0.3050\n",
      "| epoch 25 | iter 31 / 31 | loss 1.1877 | train_acc 0.6620 | test_acc 0.3000\n",
      "| epoch 26 | iter 31 / 31 | loss 1.1336 | train_acc 0.6560 | test_acc 0.2700\n",
      "| epoch 27 | iter 31 / 31 | loss 1.1378 | train_acc 0.6170 | test_acc 0.3250\n",
      "| epoch 28 | iter 31 / 31 | loss 1.2095 | train_acc 0.6570 | test_acc 0.3150\n",
      "| epoch 29 | iter 31 / 31 | loss 1.1551 | train_acc 0.6650 | test_acc 0.3000\n",
      "| epoch 30 | iter 31 / 31 | loss 1.0866 | train_acc 0.7100 | test_acc 0.3500\n",
      "| epoch 31 | iter 31 / 31 | loss 0.9811 | train_acc 0.7580 | test_acc 0.3700\n",
      "| epoch 32 | iter 31 / 31 | loss 0.8996 | train_acc 0.7700 | test_acc 0.3450\n",
      "| epoch 33 | iter 31 / 31 | loss 0.8530 | train_acc 0.7990 | test_acc 0.2950\n",
      "| epoch 34 | iter 31 / 31 | loss 0.7757 | train_acc 0.7870 | test_acc 0.3400\n",
      "| epoch 35 | iter 31 / 31 | loss 0.8031 | train_acc 0.8210 | test_acc 0.3300\n",
      "| epoch 36 | iter 31 / 31 | loss 0.6346 | train_acc 0.8560 | test_acc 0.3150\n",
      "| epoch 37 | iter 31 / 31 | loss 0.6644 | train_acc 0.8320 | test_acc 0.3000\n",
      "| epoch 38 | iter 31 / 31 | loss 0.6296 | train_acc 0.8600 | test_acc 0.3550\n",
      "| epoch 39 | iter 31 / 31 | loss 0.5797 | train_acc 0.8970 | test_acc 0.2950\n",
      "| epoch 40 | iter 31 / 31 | loss 0.4798 | train_acc 0.9190 | test_acc 0.3500\n",
      "| epoch 41 | iter 31 / 31 | loss 0.4343 | train_acc 0.9020 | test_acc 0.2700\n",
      "| epoch 42 | iter 31 / 31 | loss 0.4556 | train_acc 0.9350 | test_acc 0.3250\n",
      "| epoch 43 | iter 31 / 31 | loss 0.3652 | train_acc 0.9450 | test_acc 0.3000\n",
      "| epoch 44 | iter 31 / 31 | loss 0.4115 | train_acc 0.9220 | test_acc 0.3500\n",
      "| epoch 45 | iter 31 / 31 | loss 0.3995 | train_acc 0.9320 | test_acc 0.2900\n",
      "| epoch 46 | iter 31 / 31 | loss 0.3635 | train_acc 0.9140 | test_acc 0.3350\n",
      "| epoch 47 | iter 31 / 31 | loss 0.5024 | train_acc 0.8660 | test_acc 0.2700\n",
      "| epoch 48 | iter 31 / 31 | loss 0.4940 | train_acc 0.9040 | test_acc 0.3150\n",
      "| epoch 49 | iter 31 / 31 | loss 0.3815 | train_acc 0.9230 | test_acc 0.2900\n",
      "| epoch 50 | iter 31 / 31 | loss 0.3502 | train_acc 0.9450 | test_acc 0.3000\n",
      "max_test_acc : 0.37\n"
     ]
    }
   ],
   "source": [
    "from lenet import lenet\n",
    "from layers import *\n",
    "from optimizer import *\n",
    "from utils import *\n",
    "\n",
    "# モデルとoptimizerの生成\n",
    "model = AlexNet(input_dim =(3,32,32))\n",
    "optimizer = Adam(lr=learning_rate)\n",
    "\n",
    "data_size = len(x_train)\n",
    "max_iter = data_size // batch_size\n",
    "total_loss = 0\n",
    "count =0\n",
    "loss_list =[]\n",
    "train_acc_list = []\n",
    "test_acc_list = []\n",
    "\n",
    "\n",
    "for epoch in range(max_epoch):\n",
    "    idx = np.random.permutation(data_size)\n",
    "    x_train = x_train[idx]\n",
    "    t_train = t_train[idx]\n",
    "    \n",
    "    for iters in range(max_iter):\n",
    "        xx =x_train[iters*batch_size:(iters+1)*batch_size]\n",
    "        tt = t_train[iters*batch_size:(iters+1)*batch_size]\n",
    "       \n",
    "        # 勾配を求め、パラメータを更新\n",
    "        loss = model.forward(xx,tt)\n",
    "        model.backward()\n",
    "        optimizer.update(model.params,model.grads)\n",
    "\n",
    "        total_loss +=loss\n",
    "        count +=1\n",
    "        #model.accuracy(xx,tt)\n",
    "    \n",
    "        if (iters +1)%max_iter ==0:\n",
    "            avg_loss = total_loss/count\n",
    "            train_acc = model.accuracy(x_train, t_train)\n",
    "            test_acc = model.accuracy(x_test, t_test)\n",
    "            print('| epoch %d | iter %d / %d | loss %.4f | train_acc %.4f | test_acc %.4f' \n",
    "                      %(epoch + 1, iters + 1, max_iter, avg_loss, train_acc,test_acc))\n",
    "            loss_list.append(avg_loss)\n",
    "            train_acc_list.append(train_acc)\n",
    "            test_acc_list.append(test_acc)\n",
    "            total_loss, count = 0, 0\n",
    "print('max_test_acc :',max(test_acc_list))           "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " ### データ数を1/50に絞り学習した結果、検証データのaccuracy37%で訓練データに過学習している。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# keras-AlexNetを実装し、cifar10で学習"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "import numpy as np\n",
    "import keras\n",
    "from keras.datasets import cifar10\n",
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib import cm\n",
    "%matplotlib inline\n",
    "import random\n",
    "\n",
    "(x_train, t_train), (x_test, t_test) =cifar10.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 0〜1の範囲に正規化\n",
    "x_train = x_train.transpose(0,3,1,2)\n",
    "x_test = x_test.transpose(0,3,1,2)\n",
    "x_train = x_train.astype('float32')\n",
    "x_test = x_test.astype('float32')\n",
    "# x_train /= 255\n",
    "# x_test /= 255\n",
    "# データ数1/10で学習\n",
    "x_train, t_train = x_train[:1000], t_train[:1000]\n",
    "x_test, t_test = x_test[:200], t_test[:200]\n",
    "\n",
    "num_classes = 10\n",
    "t_train = keras.utils.to_categorical(t_train, num_classes)\n",
    "t_test = keras.utils.to_categorical(t_test, num_classes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import  Sequential\n",
    "\n",
    "from keras.layers import *\n",
    "\n",
    "model = Sequential()\n",
    "\n",
    "model.add(Conv2D(96, kernel_size=(11, 11),\n",
    "                 strides=(4, 4),\n",
    "                 padding='same',\n",
    "                 activation='relu',\n",
    "                 input_shape=(3,32,32)))\n",
    "model.add(MaxPooling2D(pool_size=(3, 3),\n",
    "                      strides=(2,2),\n",
    "                      padding='same'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(Conv2D(256, kernel_size=(5, 5),\n",
    "                 strides=(1, 1),\n",
    "                 padding='same',\n",
    "                 activation='relu'))\n",
    "model.add(MaxPooling2D(pool_size=(3, 3),\n",
    "                      strides=(2,2),\n",
    "                      padding='same'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(Conv2D(384, kernel_size=(3, 3),\n",
    "                 strides=(1, 1),\n",
    "                 padding='same',\n",
    "                 activation='relu'))\n",
    "model.add(Conv2D(384, kernel_size=(3, 3),\n",
    "                 strides=(1, 1),\n",
    "                 padding='same',\n",
    "                 activation='relu'))\n",
    "model.add(Conv2D(256, kernel_size=(3, 3),\n",
    "                 strides=(1, 1),\n",
    "                 padding='same',\n",
    "                 activation='relu'))\n",
    "model.add(MaxPooling2D(pool_size=(3, 3),\n",
    "                      strides=(2,2),\n",
    "                      padding='same'))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(2048, activation='relu'))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(2048, activation='relu'))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(num_classes, activation='softmax'))\n",
    "\n",
    "model.compile(loss=keras.losses.categorical_crossentropy,\n",
    "              optimizer=keras.optimizers.Adadelta(),\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 2.7790 - acc: 0.1410\n",
      "Epoch 2/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 2.3587 - acc: 0.1960\n",
      "Epoch 3/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 2.2160 - acc: 0.1910\n",
      "Epoch 4/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 2.1433 - acc: 0.2100\n",
      "Epoch 5/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 2.0883 - acc: 0.2190\n",
      "Epoch 6/50\n",
      "1000/1000 [==============================] - 9s 9ms/step - loss: 2.0435 - acc: 0.2330\n",
      "Epoch 7/50\n",
      "1000/1000 [==============================] - 9s 9ms/step - loss: 2.0124 - acc: 0.2370\n",
      "Epoch 8/50\n",
      "1000/1000 [==============================] - 9s 9ms/step - loss: 1.9689 - acc: 0.2490\n",
      "Epoch 9/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.9423 - acc: 0.2660\n",
      "Epoch 10/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.9098 - acc: 0.3040\n",
      "Epoch 11/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.9017 - acc: 0.2650\n",
      "Epoch 12/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.8874 - acc: 0.2920\n",
      "Epoch 13/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.8000 - acc: 0.3020\n",
      "Epoch 14/50\n",
      "1000/1000 [==============================] - 11s 11ms/step - loss: 1.8241 - acc: 0.3040\n",
      "Epoch 15/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.7882 - acc: 0.3080\n",
      "Epoch 16/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.7083 - acc: 0.3470\n",
      "Epoch 17/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.6458 - acc: 0.3840\n",
      "Epoch 18/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.6322 - acc: 0.3600\n",
      "Epoch 19/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.6034 - acc: 0.3930\n",
      "Epoch 20/50\n",
      "1000/1000 [==============================] - 9s 9ms/step - loss: 1.5596 - acc: 0.3910\n",
      "Epoch 21/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.5150 - acc: 0.4220\n",
      "Epoch 22/50\n",
      "1000/1000 [==============================] - 9s 9ms/step - loss: 1.3772 - acc: 0.4890\n",
      "Epoch 23/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.3713 - acc: 0.4790\n",
      "Epoch 24/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.3192 - acc: 0.4910\n",
      "Epoch 25/50\n",
      "1000/1000 [==============================] - 9s 9ms/step - loss: 1.3855 - acc: 0.4630\n",
      "Epoch 26/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.2604 - acc: 0.5120\n",
      "Epoch 27/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.2025 - acc: 0.5300\n",
      "Epoch 28/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.1807 - acc: 0.5710\n",
      "Epoch 29/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.1729 - acc: 0.5550\n",
      "Epoch 30/50\n",
      "1000/1000 [==============================] - 8s 8ms/step - loss: 1.1027 - acc: 0.5930\n",
      "Epoch 31/50\n",
      "1000/1000 [==============================] - 9s 9ms/step - loss: 1.1080 - acc: 0.5830\n",
      "Epoch 32/50\n",
      "1000/1000 [==============================] - 9s 9ms/step - loss: 1.0500 - acc: 0.6180\n",
      "Epoch 33/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.0781 - acc: 0.5990\n",
      "Epoch 34/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 1.0490 - acc: 0.6300\n",
      "Epoch 35/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 0.9249 - acc: 0.6560\n",
      "Epoch 36/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 0.8420 - acc: 0.7010\n",
      "Epoch 37/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 0.8580 - acc: 0.6750\n",
      "Epoch 38/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 0.7867 - acc: 0.7110\n",
      "Epoch 39/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 0.8439 - acc: 0.6860\n",
      "Epoch 40/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 0.7630 - acc: 0.7360\n",
      "Epoch 41/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 0.7915 - acc: 0.7080\n",
      "Epoch 42/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 0.8446 - acc: 0.7040\n",
      "Epoch 43/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 0.7439 - acc: 0.7400\n",
      "Epoch 44/50\n",
      "1000/1000 [==============================] - 9s 9ms/step - loss: 0.7002 - acc: 0.7660\n",
      "Epoch 45/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 0.6490 - acc: 0.7620\n",
      "Epoch 46/50\n",
      "1000/1000 [==============================] - 9s 9ms/step - loss: 0.6565 - acc: 0.7650\n",
      "Epoch 47/50\n",
      "1000/1000 [==============================] - 9s 9ms/step - loss: 0.5204 - acc: 0.8230\n",
      "Epoch 48/50\n",
      "1000/1000 [==============================] - 9s 9ms/step - loss: 0.5193 - acc: 0.8210\n",
      "Epoch 49/50\n",
      "1000/1000 [==============================] - 9s 9ms/step - loss: 0.5150 - acc: 0.8230\n",
      "Epoch 50/50\n",
      "1000/1000 [==============================] - 10s 10ms/step - loss: 0.5741 - acc: 0.8100\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0xb25e14a20>"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train, t_train,\n",
    "          epochs=50,\n",
    "          batch_size=32)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "200/200 [==============================] - 0s 987us/step\n",
      "acc: 31.00%\n"
     ]
    }
   ],
   "source": [
    "# テストデータで検証\n",
    "score = model.evaluate(x_test, t_test,batch_size=32)\n",
    "print(\"%s: %.2f%%\" % (model.metrics_names[1], score[1] * 100))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " ### 訓練データでaccuracy81%、検証データではaccuracy31%で、スクラッチ版同様、訓練データに過学習した結果になった。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
