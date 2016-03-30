import theano
#import theano.tensor as th

from solvers import SGD

def compile(weights, inputs, outputs, truths, err, extras, solver=SGD(0.01)):
    return Model(weights, inputs, outputs, truths, err, extras, solver=solver)
    

class Model(object):
    def __init__(self, weights, inputs, outputs, thruths, err, extras
                 , solver=SGD(0.01)):
        self.solver = solver
        self.weights = weights
        self.inputs = inputs
        self.truths = truths
        self.err = err
        self.extras = extras
        self.predict_ = theano.function(inputs, outputs)
        self.validate_ = theano.function(inputs+truths, [err]+extras)

    def predict(self, data):
        for args in data:
            yield self.predict_(*args)

    def validate(self, data):
        n = len(data)
        i = 0
        for args in data:
            i += 1.0
            yield i/n, self.validate_(*args)
        
    def fit(self, data):
        return self.solver(data, self.weights, self.inputs+self.truths, self.err, self.extras)
