# THEANO NN utils
import numpy as np
import theano
import theano.tensor as T
import theano.sparse as TS

def init_weights_T(*shape):
    return theano.shared((np.random.randn(*shape) * 0.01 ).astype(theano.config.floatX))


def init_weights_one_T(*shape):
    return theano.shared((np.random.randn(*shape) * 0.01 + 1.0).astype(theano.config.floatX))


def conv2D_keep_shape(x, w, image_shape, filter_shape, subsample=(1, 1)):
    # crop output to same size as input
    fs = T.shape(w)[2] - 1  # this is the filter size minus 1
    ims = T.shape(x)[2]  # this is the image size
  #  return theano.sandbox.cuda.dnn.dnn_conv(img=x, kerns=w,
    return theano.tensor.nnet.conv2d(x,w, 
					image_shape=image_shape, filter_shape=filter_shape,
                                            border_mode='full',
                                            subsample=subsample,
                                            )[:, :, fs/2:ims+fs/2, fs/2:ims+fs/2]


def rmsprop_updates_T(cost, params, stepsize=0.001, rho=0.9, epsilon=1e-6):
    # rmsprop in Theano
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - stepsize * g))
    return updates


def flip_filter(w):
    if w.ndim == 4:
        t = w.copy()
        s = t.shape
        for i in range(0, s[0]):
            for j in range(0, s[1]):
                t[i][j] = np.fliplr(t[i][j])
                t[i][j] = np.flipud(t[i][j])
        return t
    else:
        return w
