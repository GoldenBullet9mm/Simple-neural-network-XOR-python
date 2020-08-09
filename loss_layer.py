class LossLayer:

    def __init__(self):
        self.output_layer = 0

    def forward(self, input):

        self.output_layer = input
        return self.output_layer

    def backward (self, input): 
            
        loss = self.output_layer - input
        return loss