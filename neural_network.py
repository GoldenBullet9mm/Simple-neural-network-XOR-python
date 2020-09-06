import sys
import numpy as np
from decimal import Decimal

def forward(in_data, layers):
    for i in range(len(layers)):
        train_out = layers[i].forward(in_data)
        in_data = train_out
    return train_out

def backpropagation(in_data, layers):
    for b in layers[::-1]:
        data_out = b.backward(in_data)
        in_data = data_out

def mse_loss(label, prediction):
    return np.mean((label - prediction) ** 2)

def write_statistic(label, prediction, layers, iter, epoch):   

    train_out = forward(prediction, layers)
    iter += 1   
    loss = mse_loss(label, train_out)                     
    percent =  100 * iter / epoch
    sys.stdout.write("\rProgress: {}%, loss: {}".format
    (str(Decimal(percent).quantize(Decimal('.1'))),
     str(Decimal(loss).quantize(Decimal('.000001')))))

class Network:
    def __init__(self):
        self.layers = []
    def add(self, layer):
        self.layers.append(layer)
             
    def train(self, train_data, label, batch_size, epoch):
        num_batch = train_data.shape[0]
        self.label = label      
        for iteration in range (epoch):                                             
            for batch_iteration in range (0, num_batch, batch_size):
            
                if batch_iteration + batch_size < num_batch: 
                                      
                    inputs = train_data[batch_iteration: batch_iteration + batch_size]
                    forward(inputs, self.layers)
                                                                                       
                    targets = label[batch_iteration: batch_iteration + batch_size]                                
                    backpropagation(targets, self.layers)

                else:

                    inputs = train_data[batch_iteration: num_batch]
                    forward(inputs, self.layers)
                                                                       
                    targets = label[batch_iteration: num_batch]                                
                    backpropagation(targets, self.layers)
            write_statistic(label, train_data, self.layers, iteration, epoch)
           
    def prediction (self, input_data):

        data_out = forward(input_data, self.layers)        
        num_package = 0
        print('\n')
        for i in range(len(data_out)):
            num_package +=1         
            print("{})XOR:{}, expected result:{}, prediction:{}".
                  format(num_package, input_data[i], self.label[i], data_out[i]))