import numpy as np

class DefaultError:
    def init(self, shape, dtype=np.float64):
        self.dy = np.random.randn(*shape).astype(dtype)
        return self

    def __call__(self, yh):
        return np.sum(yh * self.dy)

    def grad(self, yh):
        return self.dy

def layer(layer, ins=None, outs=None, b=5, n=10, init_x=np.random.randn, type_x=None
          , err_f=DefaultError(), float_t=np.float64):

    if type_x is None:
        type_x = float_t
    
    eps = 1e-4 if float_t == np.float32 else 1e-7
    
    def num_grad( f, x ):
        dx = np.zeros(x.shape,dtype=float_t)
        y = f(x)
        for i in xrange(len(x.flat)):
            v = x.flat[i]
            x.flat[i] += eps
            yp = f(x)
            x.flat[i] = v
            dx.flat[i] = (yp - y)/eps
        return dx

    if ins is None:
        ins = layer.inputs
    if outs is None:
        outs = layer.outputs
    batches = b

    
    x = init_x(n, batches, ins).astype(type_x)
    w = layer.weights(dtype=float_t)
    dw = None if w is None else np.zeros_like(w)

    print "Testing with float_t:", float_t
    print "Testing with type_x:", type_x
    
    print "Running forward..."
    yiter, g = layer(w, [x], gradient=True)
    yh = next(yiter)
    #print yh
    
    error_f = err_f.init(yh.shape, dtype=float_t)
    dy = error_f.grad(yh)
    
    print "Running backward..."
    dx = next((a for a in g(dw, [dy])), None)

    def err_x(x):
        yh_ = next(layer(w, [x]))
        return error_f(yh_)

    def err_w(w):
        yh_ = next(layer(w, [x]))
        return error_f(yh_)

    if not dx is None:
        num_dx = num_grad(err_x, x)
        print dx
        print num_dx
        print "dX allclose:", np.allclose(dx,num_dx)

    if not dw is None:
        num_dw = num_grad(err_w, w)
        print dw
        print num_dw
        print "dW allclose:", np.allclose(dw, num_dw)

    print "Testing batching:"
    if len(x.shape) == 2:
        x_batch = np.zeros((x.shape[0], 1), dtype=x.dtype)
    else:
        x_batch = np.zeros((x.shape[0], 1, x.shape[2]), dtype=x.dtype)
    yh_per_batch = np.zeros_like(yh)
    for i in xrange(batches):
        if len(x.shape) == 2:
            x_batch[:] = x[:,i:i+1]
        else:
            x_batch[:] = x[:,i:i+1,:]
        yh_per_batch[:,i:i+1,:] = next(layer(w, [x_batch]))
    #print yh
    #print yh_per_batch
    print "Batching allclose:", np.allclose(yh, yh_per_batch)

    print "Testing steps:"
    xiter = (x[i:i+1] for i in xrange(n))
    yhiter = layer(w, xiter)
    yh_per_step = np.zeros_like(yh)
    i = 0
    for yhi in yhiter:
        yh_per_step[i:i+1] = yhi
        i += 1
    print "Steps allclose:", np.allclose(yh, yh_per_step)
