import numpy as np
from neural_network import Network
from dense_layer import DenseLayer
from sigmoid_layer import SigmoidLayer
from loss_layer import LossLayer

                              
inputs = np.array ([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]])
expected_result = np.array([[0, 1, 1, 1, 1, 0]]).T

learning_rate = 2.5
model = Network()

model.add(DenseLayer(16, 3, learning_rate))
model.add(SigmoidLayer())

model.add(DenseLayer(1, 16, learning_rate))
model.add(SigmoidLayer())

model.add(LossLayer())
model.train(inputs, expected_result, epoch = 3000)
model.prediction(inputs)