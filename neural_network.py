import sys
import numpy as np
from decimal import Decimal

class Network:
    def __init__(self):
        self.layers = []
    def add(self, layer):
        self.layers.append(layer)
     
    def train(self, in_data, expected_result, epoch):
             
        for iteration in range (epoch):
            inputs = in_data
            for i in range(len(self.layers)):
                self.data_out = self.layers[i].forward(inputs)
                inputs = self.data_out

            loss = np.mean((self.data_out - expected_result) ** 2)
            percent = 100 * iteration / epoch
            sys.stdout.write("\rProgress: {}%, loss: {}".format
            (str(percent),str(Decimal(loss).quantize(Decimal('.000001')))))
                          
            self.targets = expected_result                                
            for b in self.layers[::-1]:
                data_out = b.backward(self.targets)
                self.targets = data_out

    def prediction (self, input_data):

        for i in range(len(self.layers)):
                data_out = self.layers[i].forward(input_data)
                input_data = data_out
        print("\n\n Prediction: ")
        print(data_out)