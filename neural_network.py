import sys
import numpy as np
from decimal import Decimal

def mse_loss(label, prediction):
    return np.mean((label - prediction) ** 2)

def write_statistic(label, prediction, iter, epoch):

    iter += 1   
    loss = mse_loss(label, prediction)                     
    percent =  100 * iter / epoch
    sys.stdout.write("\rProgress: {}%, loss: {}".format
    (str(Decimal(percent).quantize(Decimal('.1'))),
     str(Decimal(loss).quantize(Decimal('.000001')))))

class Network:
    def __init__(self):
        self.layers = []
    def add(self, layer):
        self.layers.append(layer)
             
    def train(self, in_data, expected_result, epoch):
             
        for iteration in range (epoch):

            inputs = in_data
            for i in range(len(self.layers)):
                train_out = self.layers[i].forward(inputs)
                inputs = train_out

            write_statistic(expected_result, train_out, iteration, epoch)
                              
            targets = expected_result                                
            for b in self.layers[::-1]:
                data_out = b.backward(targets)
                targets = data_out
           
    def prediction (self, input_data):

        for i in range(len(self.layers)):
                data_out = self.layers[i].forward(input_data)
                input_data = data_out
        print("\n\n Prediction: ")
        print(data_out)

