import numpy as np

class LSTM(object):

    def __init__(self, ins, outs, init):
        
        pass

    # weights -> s0 -> y0 -> x -> (s,y)
    def forward(self, w, x, y, s):
        
        pass
    

class Fold(object):

    def __init__(self, layer):
        self.layer = layer

    def forward(self, w, xs, y, s):
        for x in xs:
            s0, y0 = self.layer.forward(w, s0, y0, x)
        return s0, y0


    
