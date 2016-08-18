import numpy as np
import theano
import theano.sandbox.cuda
import theano.tensor as T
import six

def as_numpy(x):
    if type(x) is theano.sandbox.cuda.CudaNdarray:
        return x
    if type(x).__module__ == np.__name__:
        return x
    return np.array(x)

def as_type(x, vtype):
    if vtype:
        return (x,)
    if type(x) is theano.sandbox.cuda.CudaNdarray:
        return x.ndim, x.dtype, 'gpu'
    return x.ndim, x.dtype, 'cpu'

def as_variable(x, name=None):
    if len(x) == 1: #x is vtype
        return x[0]
    ndim, dtype, device = x
    if device == 'gpu':
        if ndim == 0:
            return theano.sandbox.cuda.scalar(name=name,dtype=dtype)
        elif ndim == 1:
            return theano.sandbox.cuda.vector(name=name,dtype=dtype)
        elif ndim == 2:
            return theano.sandbox.cuda.matrix(name=name,dtype=dtype)
        elif ndim == 3:
            return theano.sandbox.cuda.tensor3(name=name,dtype=dtype)
        elif ndim == 4:
            return theano.sandbox.cuda.tensor4(name=name,dtype=dtype)
        raise ValueError("Type '{}' not yet supported".format(x))
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


def type_signature(args, kwargs, fixed_vtypes, kw_vtypes):
    # check whether this function is already part of a theano graph
    for x in args:
        if type(x).__module__.startswith('theano.tensor'):
            return None
    for x in six.itervalues(kwargs):
        if type(x).__module__.startswith('theano.tensor'):
            return None
    # make sure all types are numpy if they aren't value types
    args = [as_numpy(x) if not vtype else x for x,vtype in zip(args,fixed_vtypes)]
    kwargs = {k:as_numpy(v) if not k in kw_vtypes else v for k,v in six.iteritems(kwargs)}
    # make type signature
    args = [as_type(x, vtype) for x,vtype in zip(args, fixed_vtypes)]
    kwargs = {k:as_type(v, k in kw_vtypes) for k,v in six.iteritems(kwargs)}
    return tuple(args), tuple(six.iteritems(kwargs))

def make_argslist(args, kwargs, fixed_vtypes, kw_vtypes):
    args = [x for x,vt in zip(args,fixed_vtypes) if not vt]
    kwargs = [v for k,v in six.iteritems(kwargs) if k not in kw_vtypes]
    return args+kwargs

def build_graph(f, sign, fixed_vtypes, kw_vtypes, names=None, self=None):
    args, kwargs = sign 
    if names is not None:
        args = [as_variable(x, name=k) for x,k in zip(args,names)]
    else:
        args = [as_variable(x) for x in args]
    kwargs = {k:as_variable(v, name=k) for k,v in kwargs}
    args_list = make_argslist(args, kwargs, fixed_vtypes, kw_vtypes)
    if self is None:
        return f(*args, **kwargs), args_list
    else:
        return f(self, *args, **kwargs), args_list

def theano_compile(updates=False, class_method=False, print_graph_prefix=None, value_types=[]
                    , mode=theano.compile.mode.Mode()):
    def theano_compile_inner(f):
        table = {} #dispatch table
        import inspect
        argsnames, argslist, _, _ = inspect.getargspec(f)
        if class_method:
            argsnames = argsnames[1:]
        fixed_vtypes = [argsnames[i] in value_types for i in range(len(argsnames))]
        def dispatch(*args, **kwargs):
            self = None
            if class_method:
                self = args[0]
                args = args[1:]
            sign = type_signature(args, kwargs, fixed_vtypes, value_types)
            if sign is None: #this is already part of a theano graph, just call and return
                if self is None:
                    return f(*args, **kwargs)
                else:
                    return f(self, *args, **kwargs)
            #removed value types from args and kwargs
            args = [x for x,vt in zip(args, fixed_vtypes) if not vt]
            kwargs = {k:v for k,v in six.iteritems(kwargs) if not k in value_types}

            #retrieve specialized function from dispatch table
            if sign in table:
                return table[sign](*args, **kwargs)
            names = []
            for i in range(len(args)):
                name = argsnames[i] if i < len(argsnames) else argslist+str(i)
                names.append(name)
            y, x = build_graph(f, sign, fixed_vtypes, value_types, names=names, self=self)
            #from theano.compile.nanguardmode import NanGuardMode
            #mode = NanGuardMode(nan_is_error=True)
            if updates:
                y, update = y
                theano_f = theano.function(x, y, updates=update)
                #theano_f = theano.function(x, y, updates=update, mode=mode)
            else:
                theano_f = theano.function(x, y)
                #theano_f = theano.function(x, y, mode=mode)
            table[sign] = theano_f
            if print_graph_prefix is not None:
                with open(print_graph_prefix+'.txt', 'w') as out:
                    theano.printing.debugprint(theano_f, file=out)
                #theano.printing.pydotprint(theano_f, outfile=print_graph_prefix+'.png'
                #                            , var_with_name_simple=True)
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






