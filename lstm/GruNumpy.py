import numpy as np
from utils import *
from Crypto.Util.number import size

class GruNumpy:
    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1, layer_num=1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.layer_num = layer_num
        # Initialize the network parameters
        self.E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        #index:0->z,1->r,2->h
        self.U= np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (layer_num, 3, hidden_dim, hidden_dim))
        #index:0->z,1->r,2->h
        self.W= np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (layer_num, 3, hidden_dim, hidden_dim))
        self.V= np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.b = np.zeros((layer_num, 3, hidden_dim))
        #bias of output layer
        self.c = np.zeros(word_dim)
        
        print "E:" + str(self.E.shape) 
        print "U:"+ str(self. U.shape)
        print "W:"+ str(self.W.shape) 
        print "V:"+ str(self.V.shape) 
        print "b:"+ str(self.b.shape)
        print "c:"+ str(self.c.shape)
        
                # SGD / rmsprop: Initialize parameters
        self.mE = np.zeros(self.E.shape)
        self.mU = np.zeros(self.U.shape)
        self.mV = np.zeros(self.V.shape)
        self.mW = np.zeros(self.W.shape)
        self.mb = np.zeros(self.b.shape)
        self.mc = np.zeros(self.c.shape)
        
    def predict(self, x):
        o, _, _, _, _ = self.forward_propagation(x)
        return o
    
    def predict_word(self, x):
        next_word_probs = self.predict(x)
        prediction = np.argmax(next_word_probs, axis=1)
        return prediction
    
    def forward_propagation(self, x):
        
        T = len(x)
        o = np.zeros((T, self.word_dim))
        # Word embedding layer
#         print "input x:" + str(len(x)) 
        x = self.E[:,x]
#         print "embedding x:" + str(x.shape)

        s = np.zeros((T, self.layer_num, self.hidden_dim))
        z = np.zeros((T, self.layer_num, self.hidden_dim))
        r = np.zeros((T, self.layer_num, self.hidden_dim))
        h = np.zeros((T, self.layer_num, self.hidden_dim))
        for t in np.arange(T):
            ix = x[:,t]
            for l in np.arange(0, self.layer_num):
                z[t][l] = hard_sigmoid(self.U[l][0].dot(ix) + self.W[l][0].dot(s[t-1][l]) + self.b[l][0])
                r[t][l] = hard_sigmoid(self.U[l][1].dot(ix) +self. W[l][1].dot(s[t-1][l]) + self.b[l][1])
                h[t][l] = np.tanh(self.U[l][2].dot(ix) + self.W[l][2].dot(s[t-1][l]) + self.b[l][2])
                s[t][l] = (1 - z[t][l]) * h[t][l] + z[t][l] * s[t-1][l]
                ix = s[t][l]
            
            o[t] = softmax(self.V.dot(s[t][-1]) + self.c)
        
        return o, s, z, r, h
    
    def bptt(self, x, y):
        
        word = x
        T = len(y)
        o, s, z, r, h = self.forward_propagation(word)
        x = self.E[:,word];
        
        dLdE = np.zeros(self.E.shape)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        dLdb = np.zeros(self.b.shape)
        dLdc = np.zeros(self.c.shape)
        #self.bptt_truncate
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.

        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t][0].T)
            dLdc += delta_o[t]
            delta_x = 0
            for l in np.arange(self.layer_num)[::-1]:
                # Initial delta calculation
                delta_ho = self.V.T.dot(delta_o[t])
                
                for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                    
                    delta_func = map(delta_hard_sigmoid, z[bptt_step][l])
                    delta_hz =  delta_ho * (s[bptt_step - 1][l] - h[bptt_step][l]) * delta_func
                    delta_hh = delta_ho * (1 - z[bptt_step][l]) * (1-(h[bptt_step][l]**2))
                    delta_hr = self.W[l][1].T.dot(delta_hh * s[bptt_step - 1][l])
                    delta_x =  self.U[l][0].T.dot(delta_hz) + self.U[l][1].T.dot(delta_hr) + self.U[l][2].T.dot(delta_hh)
                    
                    #dW^z
                    dLdW[l][0] += delta_hz * s[bptt_step-1][l]
                    #dW^r
                    dLdW[l][1] += delta_hr * s[bptt_step-1][l]
                    #dW^h
                    dLdW[l][2] += delta_hh * s[bptt_step-1][l] * r[bptt_step][l]
    
                    input =  (x[:, bptt_step] if l < 1 else (s[bptt_step][l - 1]))
                    #dU^z
                    dLdU[l][0] += delta_hz * input
                    #dU^r
                    dLdU[l][1] += delta_hr * input
                    #dU^h
                    dLdU[l][2] += delta_hh * input
                    
                    #db^z
                    dLdb[l][0] += delta_hz
                    #db^r
                    dLdb[l][1] += delta_hr
                    #db^h
                    dLdb[l][2] += delta_hh
                    
                    #while is the first hidden layer,you need to count the dE
                    if l < 1:
                        dLdE[:, word[t]] += delta_x
                    
                    delta_ho = self.W[l][0].T.dot(delta_hz) +  self.W[l][1].T.dot(delta_hr) +  self.W[l][2].T.dot(delta_hh * r[bptt_step][l]) + delta_ho * z[bptt_step][l]

                
                
        return [dLdE, dLdU, dLdV, dLdW, dLdb, dLdc]
    
    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate, decay=0.9):
        # Calculate the gradients
        dE, dU, dV, dW, db, dc = self.bptt(x, y)

        
                # rmsprop cache updates
        self.mE = decay * self.mE + (1 - decay) * dE ** 2
        self.mU = decay * self.mU + (1 - decay) * dU ** 2
        self.mW = decay * self.mW + (1 - decay) * dW ** 2
        self.mV = decay * self.mV + (1 - decay) * dV ** 2
        self.mb = decay * self.mb + (1 - decay) * db ** 2
        self.mc = decay * self.mc + (1 - decay) * dc ** 2
        
        dE = dE / np.sqrt(self.mE + 1e-6)
        dU = dU / np.sqrt(self.mU + 1e-6)
        dV = dV / np.sqrt(self.mV + 1e-6)
        dW = dW / np.sqrt(self.mW + 1e-6)
        db = db / np.sqrt(self.mb + 1e-6)
        dc = dc / np.sqrt(self.mc + 1e-6)
        
                # Change parameters according to gradients and learning rate
        self.E -= learning_rate * dE
        self.U -= learning_rate * dU
        self.V -= learning_rate * dV
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        self.c -= learning_rate * dc
        
    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, _, _, _, _ = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        N = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(N)