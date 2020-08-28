import numpy as np

class ReLuLayer:

    def __init__(self):
        self.cache = 0

    def forward(self, input):
        
        relu = np.maximum(0, input)
        self.cache = relu
        return relu

    def backward(self, input):

        relu_Prime = np.where(self.cache > 0, 1.0, 0.0)
        return input * relu_Prime