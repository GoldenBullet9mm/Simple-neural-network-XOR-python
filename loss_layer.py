class LossLayer:

    def __init__(self):
        self.cache = 0

    def forward(self, input):

        self.cache = input
        return self.cache

    def backward (self, input):          
        return self.cache - input