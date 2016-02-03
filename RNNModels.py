# adapted from http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/LSTM.php

import numpy as np
import theano
import theano.tensor as T

dtype=theano.config.floatX

# squashing of the gates should result in values between 0 and 1
# therefore we use the logistic function
sigma = lambda x: 1 / (1 + T.exp(-x))

# for the other activation function we use the tanh
act = T.tanh

# sequences: x_t
# prior results: h_tm1, c_tm1
# non-sequences: W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xy, W_hy, W_cy, b_y
def one_lstm_step(x_t, h_tm1, c_tm1, W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y):
    i_t = sigma(theano.dot(x_t, W_xi) + theano.dot(h_tm1, W_hi) + theano.dot(c_tm1, W_ci) + b_i)
    f_t = sigma(theano.dot(x_t, W_xf) + theano.dot(h_tm1, W_hf) + theano.dot(c_tm1, W_cf) + b_f)
    c_t = f_t * c_tm1 + i_t * act(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c)
    o_t = sigma(theano.dot(x_t, W_xo)+ theano.dot(h_tm1, W_ho) + theano.dot(c_t, W_co)  + b_o)
    h_t = o_t * act(c_t)
    y_t = sigma(theano.dot(h_t, W_hy) + b_y)
    return [h_t, c_t, y_t]

#TODO: Use a more appropriate initialization method
def sample_weights(sizeX, sizeY):
    values = np.ndarray([sizeX, sizeY], dtype=dtype)
    for dx in xrange(sizeX):
        vals = np.random.uniform(low=-1., high=1.,  size=(sizeY,))
        #vals_norm = np.sqrt((vals**2).sum())
        #vals = vals / vals_norm
        values[dx,:] = vals
    _,svs,_ = np.linalg.svd(values)
    #svs[0] is the largest singular value
    values = values / svs[0]
    return values

def lstm_costfcn(n_in, n_hidden, n_i, n_c, n_o, n_f, n_y):
    # initialize weights
    # i_t and o_t should be "open" or "closed"
    # f_t should be "open" (don't forget at the beginning of training)
    # we try to archive this by appropriate initialization of the corresponding biases

    W_xi = theano.shared(sample_weights(n_in, n_i))
    W_hi = theano.shared(sample_weights(n_hidden, n_i))
    W_ci = theano.shared(sample_weights(n_c, n_i))
    b_i = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_i)))
    W_xf = theano.shared(sample_weights(n_in, n_f))
    W_hf = theano.shared(sample_weights(n_hidden, n_f))
    W_cf = theano.shared(sample_weights(n_c, n_f))
    b_f = theano.shared(np.cast[dtype](np.random.uniform(0, 1.,size = n_f)))
    W_xc = theano.shared(sample_weights(n_in, n_c))
    W_hc = theano.shared(sample_weights(n_hidden, n_c))
    b_c = theano.shared(np.zeros(n_c, dtype=dtype))
    W_xo = theano.shared(sample_weights(n_in, n_o))
    W_ho = theano.shared(sample_weights(n_hidden, n_o))
    W_co = theano.shared(sample_weights(n_c, n_o))
    b_o = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_o)))
    W_hy = theano.shared(sample_weights(n_hidden, n_y))
    b_y = theano.shared(np.zeros(n_y, dtype=dtype))

    c0 = theano.shared(np.zeros(n_hidden, dtype=dtype))
    h0 = T.tanh(c0)

    params = [W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y, c0]

    #input
    v = T.matrix(dtype=dtype)

    # target
    target = T.matrix(dtype=dtype)

    # hidden and outputs of the entire sequence
    [h_vals, _, y_vals], _ = theano.scan(fn=one_lstm_step,
                                         sequences = dict(input=v, taps=[0]),
                                         outputs_info = [h0, c0, None ], # corresponds to return type of fn
                                         non_sequences = [W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc,
                                                          b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y] )

    # cost function
    # cost = -T.mean(target * T.log(y_vals)+ (1.- target) * T.log(1. - y_vals))
    cost = - T.mean((target-y_vals)*(target-y_vals))

    # learning rate
    lr = np.cast[dtype](.1)
    learning_rate = theano.shared(lr)

    # calculating gradients
    gparams = []
    for param in params:
      gparam = T.grad(cost, param)
      gparams.append(gparam)

    updates=[]
    for param, gparam in zip(params, gparams):
        updates.append((param, param - gparam * learning_rate))

    learn_rnn_fn = theano.function(inputs = [v, target],
                                   outputs = cost,
                                   updates = updates)

    return learn_rnn_fn

