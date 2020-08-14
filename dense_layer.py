import numpy as np

class DenseLayer:

    def __init__(self, number_neurons, neuron_input, learning_rate):
              
        self.weights = 0.10 * np.random.randn(neuron_input, number_neurons)
        self.biases = np.zeros(number_neurons)
        self.output_layer = 0
        self.learning_rate = learning_rate
        self.prev_grad_w = np.zeros_like(self.weights)
        self.prev_grad_b = np.zeros_like(self.biases)
                                   
    def forward (self, inputs):

        self.output_layer = inputs      
        output = np.dot(inputs, self.weights) + self.biases        
        return output

    def backward(self, delta):
        
        momentum = 0.9
        batch_size = delta.shape[0]
                         
        data = np.dot(delta, self.weights.T) 
        gradient_weights =  np.dot (self.output_layer.T, delta) / batch_size       
        gradient_biases =  np.sum(delta, axis = 0, keepdims = False) / batch_size

        self.prev_grad_w = self.prev_grad_w * momentum + gradient_weights
        self.prev_grad_b = self.prev_grad_b * momentum + gradient_biases
                        
        self.weights -= self.learning_rate * self.prev_grad_w
        self.biases -= self.learning_rate * self.prev_grad_b

        return data