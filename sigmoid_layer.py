import numpy as np

class SigmoidLayer:

    def __init__(self):
        self.cache = 0
           
    def forward(self, input):
       
        sigmoid = 1 / (1 + np.exp (- input))
        self.cache = sigmoid
        return sigmoid

    def backward(self, input):

        sigmoid_derivative = self.cache * (1 - self.cache)
        return input * sigmoid_derivative