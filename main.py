# Neural network XOR problem.
# Created by Dmitry Lebedev on 10/08/2020.
# Copyright © 2020 Dmitry Lebedev. All rights reserved.
import numpy as np
from neural_network import Network
from dense_layer import DenseLayer
from sigmoid_layer import SigmoidLayer
from relu_layer import ReLuLayer
from loss_layer import LossLayer
                             
inputs = np.array ([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]])
expected_result = np.array([[0, 1, 1, 1, 1, 0]]).T

learning_rate = 0.9
model = Network()

model.add(DenseLayer(16, 3, learning_rate))
model.add(ReLuLayer())

model.add(DenseLayer(1, 16, learning_rate))
model.add(SigmoidLayer())
model.add(LossLayer())

model.train(inputs, expected_result, batch_size = 1, epoch = 500)
model.prediction(inputs)