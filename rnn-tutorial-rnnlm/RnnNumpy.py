#coding=utf-8
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
from utils import *

import matplotlib.pyplot as plt


class RNNNumpy:
    
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        """
        Instantiates the RNN class.

        Parameters
        ----------
        word_dim : 输入词向量的维度，如果是one hot，显然应该等于Input Layer 的结点数
        hidden_dim : 隐层的结点数
        bptt_truncate：BPTT反向传播的时间范围
        """

        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        """
        forward propagation algorithm.

        Parameters
        ----------
        x : 输入句子对应的词向量list,其长度等于句子中的word数目，每一个word的词向量作为一个时刻t的网络输入
        
        Returns
        -------
        [o, s]：
            y:每一时刻的网络输出值，其大小为T×N_o
                其中的一个元素o[t][i]含义为$p(w_{t+1}=w_i|w_t)$,即已知t时刻输入word $w_t$,的前提下，下一个单词为$w_i$的概率
            s:每一时刻Hidden Layer节点的输出值，其长度为T×N_h
                其中的一个元素s[t][i]表示t时刻隐层第i个节点的输出值
        """
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]
    
#     RNNNumpy.forward_propagation = forward_propagation

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)
    
#     RNNNumpy.predict = predict
    
    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L
    
    def calculate_loss(self, x, y):
        """
        loss function.

        Parameters
        ----------
        x : 输入句子对应的词向量list 矩阵，矩阵的行数等于输入句子的数目，而每一行存储的内容为各个句子对应的word index列表
        y : 训练值，即期望的输出word index 列表
        
        Returns
        avg loss：所有训练样本平均loss
        -------
        """
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N
    
#     RNNNumpy.calculate_total_loss = calculate_total_loss
#     RNNNumpy.calculate_loss = calculate_loss

    def bptt(self, x, y):
        """
        BPTT algorithm.

        Parameters
        ----------
        x : 输入句子对应的词向量list,其长度等于句子中的word数目，每一个word的词向量作为一个时刻t的网络输入
        y : 训练值，即期望的输出word index 列表
        
        Returns
        dLdU：损失函数对U矩阵的导数
        dLdV：损失函数对V矩阵的导数
        dLdW：损失函数对W矩阵的导数
        -------
        """
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])              
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]
    
#     RNNNumpy.bptt = bptt


    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = model.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = model.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = model.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return 
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)
    
#     RNNNumpy.gradient_check = gradient_check


    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
    
#     RNNNumpy.sgd_step = numpy_sdg_step

