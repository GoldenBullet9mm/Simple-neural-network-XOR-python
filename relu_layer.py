import numpy as np

class ReLuLayer:

    def __init__(self):
        self.output_layer = 0

    def forward(self, input):
        
        activation = np.maximum(0, input)
        self.output_layer = activation
        return activation

    def backward(self, input):

        data_in = input        
        self.output_layer[self.output_layer < 0] = 0
        delta = data_in * self.output_layer
        return delta