
def fast_tanh(x):
    return x/(1+abs(x))

def fast_sigmoid(x):
    return (fast_tanh(x)+1)/2



