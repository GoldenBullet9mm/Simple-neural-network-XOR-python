import numpy as np

class SigmoidLayer:

    def __init__(self):
        self.output_layer = 0
           
    def forward(self, input):
       
        activation = 1 / (1 + np.exp (- input))
        self.output_layer = activation
        return activation

    def backward(self, input):

        data_in = input        
        delta = data_in * self.output_layer * (1 - self.output_layer)
        return delta