import numpy as np
import theano
import theano.tensor as T
import six

def as_numpy(x):
    if type(x).__module__ == np.__name__:
        return x
    return np.array(x)

def as_type(x):
    return x.ndim, x.dtype

def as_variable(x, name=None):
    ndim, dtype = x
    if ndim == 0:
        return T.scalar(name=name,dtype=dtype)
    elif ndim == 1:
        return T.vector(name=name,dtype=dtype)
    elif ndim == 2:
        return T.matrix(name=name,dtype=dtype)
    elif ndim == 3:
        return T.tensor3(name=name,dtype=dtype)
    elif ndim == 4:
        return T.tensor4(name=name,dtype=dtype)
    raise ValueError("Type '{}' not yet supported".format(x))


def type_signature(args, kwargs, flags, kw_flags):
    # check whether this function is already part of a theano graph
    for x in args:
        if type(x).__module__.startswith('theano.tensor'):
            return None
    for x in six.itervalues(kwargs):
        if type(x).__module__.startswith('theano.tensor'):
            return None
    # make sure all types are numpy
    args = [as_numpy(x) for x in args]
    kwargs = {k:as_numpy(v) for k,v in six.iteritems(kwargs)}
    # make type signature
    args = [as_type(x) for x in args]
    kwargs = {k:as_type(v) for k,v in six.iteritems(kwargs)}
    return tuple(args), tuple(six.iteritems(kwargs))

def build_graph(f, sign, self=None):
    args, kwargs = sign 
    args = [as_variable(x) for x in args]
    kwargs = {k:as_variable(v, name=k) for k,v in kwargs}
    if self is None:
        return f(*args, **kwargs), args+list(kwargs.values())
    else:
        return f(self, *args, **kwargs), args+list(kwargs.values())

def theano_compile(updates=False, class_method=False):
    def theano_compile_inner(f):
        table = {} #dispatch table
        def dispatch(*args, **kwargs):
            self = None
            if class_method:
                self = args[0]
                args = args[1:]
            sign = type_signature(args, kwargs, [], {})
            if sign is None: #this is already part of a theano graph, just call and return
                if self is None:
                    return f(*args, **kwargs)
                else:
                    return f(self, *args, **kwargs)
            #retrieve specialized function from dispatch table
            if sign in table:
                return table[sign](*args, **kwargs)
            y, x = build_graph(f, sign, self=self)
            if updates:
                y, update = y
                theano_f = theano.function(x, y, updates=update)
            else:
                theano_f = theano.function(x, y)
            table[sign] = theano_f
            return theano_f(*args, **kwargs)
        return dispatch
    return theano_compile_inner

def get_flag(name, ignore, const):
    if name in ignore:
        return 0
    if name in const:
        return 2
    return 1

def theano_compile_v2(ignore=[], const=[]):
    import inspect
    if type(ignore) is str:
        ignore = [ignore]
    if type(const) is str:
        const = [const]
    def theano_compile(f):
        names = inspect.getargspec(f)[0]
        flags = [get_flag(name, ignore, const) for name in names]
        kw_flags = {names[i]:flags[i] for i in xrange(len(names))}
        table = {} #dispatch table
        def dispatch(*args, **kwargs):
            sign = type_signature(*args, **kwargs)
            if sign is None: #this is already part of a theano graph, just call and return
                return f(*args, **kwargs)
            #retrieve specialized function from dispatch table
            if sign in table:
                return table[sign](*args, **kwargs)
            y, x = build_graph(f, sign)
            theano_f = theano.function(x, y)
            table[sign] = theano_f
            return theano_f(*args, **kwargs)
        return dispatch






