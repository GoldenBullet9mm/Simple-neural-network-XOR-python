# Simple neural network
To train this network I wrote a library.
And so, first, let's connect the modules, and numpy library.
```
import numpy as np
from neural_network import Network
from dense_layer import DenseLayer
from sigmoid_layer import SigmoidLayer
from relu_layer import ReLuLayer
from loss_layer import LossLayer
```
Then we need to write the data on which we will train the network.
```
inputs = np.array ([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]])
expected_result = np.array([[0, 1, 1, 1, 1, 0]]).T
```
Next, we need to write the learning rate, I wrote 0.9. And we initialize our model.
```
learning_rate = 0.9
model = Network()
```
Then we build our network. We build the first layer. We write the number of neurons,
and the number of inputs and the learning rate. I wrote 16 neurons for the first layer. 
Now we need an activation function for the first layer. 
For the first layer I will use the relu function.
```
model.add(DenseLayer(16, 3, learning_rate))
model.add(ReLuLayer())
```
Building the second layer of our network. 
In the second layer we will have one neuron and 16 inputs from the first layer.
The activation function in the last layer will be sigmoid.
Since this is the last layer of our network, we need to add a 
LossLayer to calculate the difference between the output of the 
neural network and the expected result.

```
model.add(DenseLayer(1, 16, learning_rate))
model.add(SigmoidLayer())
model.add(LossLayer())
```
And so we built our neural network. Now we need to train it.
Calling the method train. 
We write our data. We write the size of the batch 1, 
and we write epochs, i will have 500 epochs.
```
model.train(inputs, expected_result, batch_size = 1, epoch = 500)
```
And we call the prediction method to see the result of the neural network.
And we write inputs.
```
model.prediction(inputs)
```
