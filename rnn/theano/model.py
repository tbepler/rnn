import theano
#import theano.tensor as th

from solvers import SGD

class Model(object):
    def __init__(self, ws, yh, x, y, err, *args, **kwargs):
        self.solver = kwargs.get('solver', SGD(0.01))
        self.weights = ws
        self.x = x
        self.y = y
        self.err = err
        self.args = args
        self.predict_ = theano.function([x], yh)
        self.validate_ = theano.function([x,y], [err]+list(args))

    def predict(self, data):
        n = len(data)
        i = 0.0
        for x in data:
            i += 1
            yield i/n, self.predict_(x)

    def validate(self, data):
        n = len(data)
        i = 0.0
        for x,y in data:
            i += 1
            yield i/n, self.validate_(x, y)

    def fit(self, data):
        return self.solver(data, self.weights, self.x, self.y, self.err, *self.args)
