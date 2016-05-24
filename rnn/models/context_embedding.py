import theano
import theano.tensor as T
import numpy as np

import rnn.theano.lstm as lstm
import rnn.theano.linear as linear

import rnn.theano.solvers as solvers

def null_func(*args, **kwargs):
    pass

class TheanoContextEmbedding(object):
    def __init__(self, encoder, loss):
        self.encoder = encoder
        self._loss = loss

    def transform(self, X):
        return self.encoder(X)

    def logprob(self, X):
        Z = self.transform(X)
        return self._loss.logprob(Z)

    def loss(self, X, Y):
        Z = self.transform(X)
        return self._loss(Z, Y)

class ContextEmbedding(object):
    def __init__(self, encoder, optimizer=solvers.RMSprop(0.01), batch_size=100, length=-1, flank=0):
        self.encoder = encoder
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.length = length
        self.flank = flank
        self.theano_setup()

    def __getstate__(self):
        state = {}
        state['encoder'] = self.encoder
        state['optimizer'] = self.optimizer
        state['batch_size'] = self.batch_size
        state['length'] = self.length
        state['flank'] = self.flank
        return state

    def __setstate__(self, state):
        self.__init__(state['encoder'], optimizer=state['optimizer']
                , batch_size=state['batch_size']
                , length=state['length']
                , flank=state['flank'])

    def theano_setup(self):
        self.theano = object()
        self.theano.X = T.matrix()
        self.theano.mask = T.matrix(dtype='int8')

    def split(self, data, length=None, flank=None):
        if length is None:
            length = self.length
        if flank is None:
            flank = self.flank
        index = 0
        for x in data:
            mask = None
            if flank > 0:
                n = len(x)
                mask = np.zeros(n+2*flank, dtype=np.int8)
                mask[flank:-flank] = 1
                x_flanked = np.zeros(n+2*flank, dtype=x.dtype)
                x_flanked[flank:-flank] = x
                x = x_flanked
            if length > 0:
                m = length + 2*flank
                for i in xrange(0, n, length):
                    if mask is not None:
                        yield index, x[i:i+m], mask[i:i+m]
                    else:
                        yield index, x[i:i+m]
            else:
                if mask is not None:
                    yield index, x, mask
                else:
                    yield index, x
            index += 1

    def make_minibatch(self, xs):
        b = len(xs)
        cols = zip(*xs)
        if len(cols) > 2:
            index, xs, mask = cols
        else:
            index, xs = cols
            mask = None
        n = max(len(x) for x in xs)
        X = np.zeros((n,b), dtype=xs[0].dtype)
        M = np.zeros((n,b), dtype=np.int8)
        for i in xrange(b):
            k = len(xs[i])
            X[:k,i] = xs[i]
            if mask is not None: 
                M[:k,i] = mask[i]
            else:
                M[:k,i] = 1
        return index, X, M

    def minibatched(self, data):
        xs = []
        for x in data:
            if len(xs) >= self.batch_size:
                yield self.make_minibatch(xs)
                del xs[:]
            xs.append(x)
        if len(xs) > 0:
            yield self.make_minibatch(xs)

    def _compile_loss_minibatch(self):
        if not hasattr(self, 'loss_minibatch'):
            X = self.theano.X
            mask = self.theano.mask
            res = self.encoder.loss(X, mask)
            if flank > 0:
                res = [r[self.flank:-self.flank] for r in res]
            res = [T.sum(r, axis=0) for r in res]
            self.loss_minibatch = theano.function([X, mask], res)  

    def loss_iter(self, data, callback=null_func):
        i = 0
        L = None
        for index, X, M in self.minibatched(self.split(data)):
            res = self.loss_minibatch(X, M)
            if L is None:
                L = [0 for _ in res]
            for j in xrange(len(index)):
                if index[j] > i:
                    yield L
                    L = [0 for _ in res]
                    i = index[j]
                L = [l+r[j] for l,r in zip(L,res)]







        f = self._compile_loss(dtype=data.dtype, devices=devices, axis=0, chunk_size=chunk_size)
        callback(0, 'loss')
        i = 0
        for X, mask in data:
            res = f(X, mask)
            i += 1
            p = float(i)/len(data)
            callback(p, 'loss')
            yield res

    def loss(self, data, callback=null_func, devices=[], axis=None, chunk_size=-1):
        if axis == 0:
            return self.loss_iter(data, callback=callback, devices=devices, chunk_size=chunk_size)
        f = self._compile_loss(dtype=data.dtype, devices=devices, axis=axis, chunk_size=chunk_size)
        callback(0, 'loss')
        cent, conf, i = 0, 0, 0
        for X, mask in data:
            cent_, conf_ = f(X, mask)
            cent += cent_
            conf += conf_
            i += 1
            p = float(i)/len(data)
            callback(p, 'loss')
        return cent, conf

    
    
class CharRNN(object):
    def __init__(self, n_in, n_out, layers, decoder=linear.LinearDecoder, solver=solvers.RMSprop(0.01)
            , l1_reg=0, cov_reg=0, unroll=-1):
        self.n_in = n_in
        self.n_out = n_out
        self.layers = layers
        self.decoder = decoder
        self.solver = solver
        self.l1_reg = l1_reg
        self.cov_reg = cov_reg
        self.unroll = unroll
        self.setup()

    def setup(self, weights=None):
        self._theano_model = CharRNNGraph(self.n_in, self.n_out, self.layers, decoder=self.decoder
                , weights=weights)

    @property
    def weights(self):
        return [w.get_value(borrow=True) for w in self._theano_model.weights]

    @property
    def parameters(self):
        params = {}
        params['n_in'] = self.n_in
        params['n_out'] = self.n_out
        params['layers'] = self.layers
        params['decoder'] = self.decoder
        params['solver'] = self.solver
        params['l1_reg'] = self.l1_reg
        params['cov_reg'] = self.cov_reg
        params['unroll'] = self.unroll
        return params
    
    def load_parameters(self, params):
        self.n_in = params['n_in']
        self.n_out = params['n_out']
        self.layers = params['layers']
        self.decoder = params.get('decoder', linear.Linear)
        self.solver = params.get('solver', solvers.RMSprop(0.01))
        self.l1_reg = params.get('l1_reg', 0)
        self.cov_reg = params.get('cov_reg', 0)
        self.unroll = params.get('unroll', -1)

    def __getstate__(self):
        state = {}
        state['weights'] = self.weights
        state['params'] = self.parameters
        return state

    def __setstate__(self, state):
        self.load_parameters(state['params'])
        self.setup(weights=state['weights'])

    def _device_models(self, devices, copy_shared=False):
        if len(devices) == 0:
            return [self._theano_model]
        return [self._theano_model.transfer(target, copy_shared=copy_shared) for target in devices]

    def _theano_crossent(self, Yh, Y, mask):
        cent = T.zeros_like(Yh)
        i,j = T.mgrid[0:cent.shape[0], 0:cent.shape[1]]
        cent = T.set_subtensor(cent[i,j,Y], -Yh[i,j,Y]*mask)
        return cent

    def _theano_confusion(self, Yh, Y, mask):
        Yh = T.argmax(Yh, axis=-1)
        shape = list(Yh.shape) + [self.n_out, self.n_out]
        C = T.zeros(shape, dtype='int64')
        i,j = T.mgrid[0:C.shape[0], 0:C.shape[1]]
        C = T.set_subtensor(C[i,j,Y,Yh], 1)
        mask = T.shape_padright(T.shape_padright(mask))
        C = C*mask
        return C

    def _theano_loss_chunked(self, X, Y, mask, model, axis, chunk_size):
        n = model.nlayers
        def _step(i, cent0, conf0, *args):
            y0 = args[:n]
            c0 = args[n:2*n]
            X = args[2*n][i:i+chunk_size]
            Y = args[2*n+1][i:i+chunk_size]
            mask = args[2*n+2][i:i+chunk_size]
            Yh, y_layers, cs = model.logprob(Y, X, c0=c0, y0=y0, unroll=self.unroll)
            cent = cent0 + T.sum(self._theano_crossent(Yh, Y, mask), axis=axis)
            conf = conf0 + T.sum(self._theano_confusion(Yh, Y, mask), axis=axis)
            ys = [y_layer[-1] for y_layer in y_layers] 
            cs = [c[-1] for c in cs]
            return [cent, conf] + ys + cs
        idxs = T.arange(0, X.shape[0], chunk_size)
        _,b = X.shape
        y0 = [T.zeros((b,layer.units)) for layer in model.layers]
        c0 = [T.zeros((b,layer.units)) for layer in model.layers]
        if axis == 0:
            loss0 = T.zeros((b,self.n_out))
            confusion0 = T.zeros((b, self.n_out, self.n_out), dtype='int64')
        else:
            loss0 = T.zeros(self.n_out)
            confusion0 = T.zeros((self.n_out, self.n_out), dtype='int64')
        res, _ = theano.foldl(_step, idxs, [loss0, confusion0]+y0+c0, non_sequences=[X, Y, mask])
        return res[0], res[1]

    def _theano_loss(self, X, Y, mask, models, axis=0, chunk_size=-1):
        idxs = [T.arange(i, X.shape[1], len(models)) for i in xrange(len(models))]
        Ys = [Y[:,idx] for idx in idxs]
        masks = [mask[:,idx] for idx in idxs]
        if axis is None:
            axis_conf = [0,1]
        else:
            axis_conf = axis
        if chunk_size > 0:
            cents = []
            confs = []
            for i in xrange(len(models)):
                cent, conf = self._theano_loss_chunked(X[:,idxs[i]], Ys[i], masks[i], models[i], axis_conf, chunk_size)
                cents.append(cent)
                confs.append(conf)
        else:
            Yhs = [model.logprob(Y[:,idx], X[:,idx], unroll=self.unroll)[0] for model,idx in zip(models, idxs)]
            cents = [T.sum(self._theano_crossent(Yh, Yi, maski), axis=axis_conf) for Yh,Yi,maski in zip(Yhs, Ys, masks)]
            confs = [T.sum(self._theano_confusion(Yh, Yi, maski), axis=axis_conf) for Yh,Yi,maski in zip(Yhs, Ys, masks)]
        if axis == 0:
            idxs = T.concatenate(idxs, axis=0)
            idx_inv = T.arange(X.shape[1])[idxs]
            cent = T.concatenate(cents, axis=0)[idx_inv]
            conf = T.concatenate(confs, axis=0)[idx_inv]
        else:
            cent = sum(cents)
            conf = sum(confs)
        return cent, conf

    def _compile_loss(self, dtype='int32', devices=[], axis=0, chunk_size=-1):
        devices = self._device_models(devices, copy_shared=True)
        data = T.matrix(dtype=dtype)
        X = data[:-1]
        Y = data[1:]
        mask = T.matrix(dtype='int8')
        results = self._theano_loss(data[:-1], data[1:], mask[1:], devices, axis=axis, chunk_size=chunk_size)
        f = theano.function([data, mask], results)
        return f

    def loss_iter(self, data, callback=null_func, devices=[], chunk_size=-1):
        f = self._compile_loss(dtype=data.dtype, devices=devices, axis=0, chunk_size=chunk_size)
        callback(0, 'loss')
        i = 0
        for X, mask in data:
            res = f(X, mask)
            i += 1
            p = float(i)/len(data)
            callback(p, 'loss')
            yield res

    def loss(self, data, callback=null_func, devices=[], axis=None, chunk_size=-1):
        if axis == 0:
            return self.loss_iter(data, callback=callback, devices=devices, chunk_size=chunk_size)
        f = self._compile_loss(dtype=data.dtype, devices=devices, axis=axis, chunk_size=chunk_size)
        callback(0, 'loss')
        cent, conf, i = 0, 0, 0
        for X, mask in data:
            cent_, conf_ = f(X, mask)
            cent += cent_
            conf += conf_
            i += 1
            p = float(i)/len(data)
            callback(p, 'loss')
        return cent, conf

    def _theano_training_loss(self, Yh, Y, mask, activations):
        loss = T.sum(self._theano_crossent(Yh, Y, mask))
        #impose an l1 penalty on the activations
        if self.l1_reg > 0:
            loss += sum(T.sum(abs(act*T.shape_padright(mask))*self.l1_reg) for act in activations)
        if self.cov_reg > 0:
            #only impose covariance regularizer on the top level activations
            n = T.sum(mask)
            act = activations[-1][mask.nonzero()]
            #act = act.reshape((act.shape[0]*act.shape[1], act.shape[2]))
            act = act/T.sqrt(T.sum(act**2, axis=0, keepdims=True))
            cov = T.dot(act.T, act)
            m = cov.shape[0]
            i = T.arange(m)
            cov = T.set_subtensor(cov[i,i], 0)
            loss += self.cov_reg*n*cov.norm(1)/(m*(m-1))
        return loss

    def _theano_gradient_truncated(self, X, Y, mask, model, backprop):
        n = model.nlayers
        def step(i, *args):
            gw0 = args[:len(model.weights)]
            loss0 = args[len(model.weights)]
            confusion0 = args[len(model.weights)+1]
            args = args[len(model.weights)+2:]
            y0 = args[:n]
            c0 = args[n:2*n]
            X = args[2*n]
            Y = args[2*n+1]
            mask = args[2*n+2]
            Yh,y_layers,cs = model.logprob(Y[i:i+backprop], X[i:i+backprop], y0=y0, c0=c0, unroll=self.unroll)
            loss = self._theano_training_loss(Yh, Y[i:i+backprop], mask[i:i+backprop], y_layers)
            constants = list(y0) + list(c0) + [i]
            gw = theano.grad(loss, model.weights, consider_constant=constants)
            gw = [gwi+gw0i for gwi,gw0i in zip(gw, gw0)]
            cs = [c[-1] for c in cs]
            ys = [y_layer[-1] for y_layer in y_layers]
            confusion = T.sum(self._theano_confusion(Yh, Y[i:i+backprop], mask[i:i+backprop]), axis=[0,1])
            loss = T.sum(self._theano_crossent(Yh, Y[i:i+backprop], mask[i:i+backprop]), axis=[0,1])
            loss += loss0
            confusion += confusion0
            return gw + [loss, confusion] + ys + cs
        idxs = T.arange(0, X.shape[0], backprop)
        _,b = X.shape
        y0 = [T.zeros((b,layer.units)) for layer in model.layers]
        c0 = [T.zeros((b,layer.units)) for layer in model.layers]
        gw0 = [T.zeros_like(w) for w in model.weights]
        loss0 = T.zeros(self.n_out)
        confusion0 = T.zeros((self.n_out, self.n_out), dtype='int64')
        res, _ = theano.foldl(step, idxs, gw0+[loss0, confusion0]+y0+c0, non_sequences=[X, Y, mask])
        gws = res[:len(model.weights)]
        loss = res[len(model.weights)]
        confusion = res[len(model.weights)+1]
        return gws, loss, confusion

    def _theano_gradient(self, X, Y, mask, models, truncated_backprop=0):
        gws = []
        losses = []
        confusions = []
        for i in xrange(len(models)):
            idx = T.arange(i, X.shape[1], len(models))
            if truncated_backprop > 0:
                gw, loss, confusion = self._theano_gradient_truncated(X[:,idx], Y[:,idx], mask[:,idx], models[i], truncated_backprop)
            else:
                model = models[i]
                Yh,y_layers,_ = model.logprob(Y[:,idx], X[:,idx], unroll=self.unroll)
                loss = self._theano_training_loss(Yh, Y[:,idx], mask[:,idx], y_layers)
                gw = theano.grad(loss, model.weights)
                confusion = T.sum(self._theano_confusion(Yh, Y[:,idx], mask[:,idx]), axis=[0,1])
                loss = T.sum(self._theano_crossent(Yh, Y[:,idx], mask[:,idx]), axis=[0,1])
            gws.append(gw)
            losses.append(loss)
            confusions.append(confusion)
        loss = sum(losses)
        confusion = sum(confusions)
        gws = zip(*gws)
        return [sum(g) for g in gws], loss, confusion

    def fit_steps(self, train, max_iters=100, truncated_backprop=-1, devices=[]):
        models = self._device_models(devices, copy_shared=False)
        data = T.matrix(dtype=train.dtype)
        mask = T.matrix(dtype='int8')
        gws, loss, confusion = self._theano_gradient(data[:-1], data[1:], mask[1:], models, truncated_backprop=truncated_backprop)
        weights = self._theano_model.weights
        return self.solver(train, [data, mask], [loss, confusion], weights, grads=gws, max_iters=max_iters)        

    def fit(self, train, validate=None, max_iters=100, callback=null_func, truncated_backprop=-1, devices=[]):
        steps = self.fit_steps(train, max_iters=max_iters, truncated_backprop=truncated_backprop, devices=devices)
        callback(0, 'fit')
        train_loss, train_confusion = 0, 0
        for it, [loss, confusion] in steps:
            train_loss += loss
            train_confusion += confusion
            if it % 1 == 0:
                if validate is not None:
                    val_loss, val_confusion = self.loss(validate, callback=callback, devices=devices, chunk_size=truncated_backprop)
                    yield val_loss, val_confusion, train_loss, train_confusion
                else:
                    yield train_loss, train_confusion
                train_loss, train_confusion = 0, 0
            callback(it%1, 'fit')
    
    def _theano_transform(self, X, model, features):
        feats,_,_ = model.transform(X, features=features, unroll=self.unroll)
        return feats

    def _compile_transform_chunks(self, data, models, chunk_size, features):
        n = self._theano_model.nlayers
        c0 = [T.matrix() for _ in xrange(n)]
        y0 = [T.matrix() for _ in xrange(n)]
        idxs = [T.arange(i, data.shape[1], len(models)) for i in xrange(len(models))]
        feats, cs, ys = [], [], []
        for idx, model in zip(idxs, models):
            f,y_layers,cs_ = model.transform(data[:,idx], c0=[c[idx] for c in c0], y0=[y[idx] for y in y0]
                    , unroll=self.unroll, features=features)
            feats.append(f)
            cs.append([c[-1] for c in cs_])
            ys.append([y_layer[-1] for y_layer in y_layers])
        inv_idx = T.arange(data.shape[1])[T.concatenate(idxs, axis=0)]
        cs = [T.concatenate(c,axis=0)[inv_idx] for c in zip(*cs)]
        ys = [T.concatenate(y,axis=0)[inv_idx] for y in zip(*ys)]
        feats = T.concatenate(feats, axis=1)[:,inv_idx]
        f = theano.function([data]+y0+c0, [feats]+ys+cs)
        def _function(X):
            k,b = X.shape
            y0 = [np.zeros((b,m), dtype=theano.config.floatX) for m in self.layers]
            c0 = [np.zeros((b,m), dtype=theano.config.floatX) for m in self.layers]
            for i in xrange(0,k,chunk_size):
                args = [X[i:i+chunk_size]]+y0+c0
                res = f(*args)
                yield res[0]
                y0 = res[1:n+1]
                c0 = res[n+1:2*n+1]
        return _function

    def _compile_transform(self, dtype, features, devices=[], chunk_size=-1):
        models = self._device_models(devices, copy_shared=True)
        data = T.matrix(dtype=dtype)
        if chunk_size > 0:
            return self._compile_transform_chunks(data, models, chunk_size, features)
        idxs = [T.arange(i, data.shape[1], len(devices)) for i in xrange(len(devices))]
        feats = [self._theano_transform(data[:,idx], model, features) for idx,model in zip(idxs,models)]
        feats = T.concatenate(feats, axis=1)
        idxs = T.concatenate(idxs, axis=0)
        inv_idx = T.arange(data.shape[1])[idxs]
        feats = feats[:, inv_idx]
        return theano.function([data], [feats])

    def transform(self, data, features=[-1], chunk_size=-1, devices=[], callback=null_func):
        f = self._compile_transform(data.dtype, features, devices=devices, chunk_size=chunk_size)
        callback(0, 'transform')
        i = 0
        for X in data:
            if X.ndim < 2:
                X = np.reshape(X, list(X.shape)+[1])
            Z = f(X)
            i += 1
            p = float(i)/len(data)
            yield Z
            callback(p, 'transform')


