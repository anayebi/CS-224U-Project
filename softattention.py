# Adapted from code written by braingineer: https://gist.github.com/braingineer/27c6f26755794f6544d83dec2dd27bbb 
# referenced in https://github.com/fchollet/keras/issues/2612

import numpy as np
import copy
import inspect
import types as python_types
import marshal
import sys
import warnings

from keras import backend as K
from keras import activations, initializations, regularizers
from keras.engine.topology import Layer, InputSpec
from keras.layers.wrappers import Wrapper, TimeDistributed
from keras.layers.core import Dense
from keras.layers.recurrent import Recurrent, time_distributed_dense

#max_sen_length = get_max_sen_len() # can use this or if you know just give it the number
max_sen_length = 56

def make_safe(x):
    return K.clip(x, K.common._EPSILON, 1.0 - K.common._EPSILON)

class ProbabilityTensor(Wrapper):
    """ function for turning 3d tensor to 2d probability matrix, which is the set of a_i's """
    def __init__(self, dense_function=None, *args, **kwargs):
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        #layer = TimeDistributed(dense_function) or TimeDistributed(Dense(1, name='ptensor_func'))
        layer = TimeDistributed(Dense(1, name='ptensor_func'))
        super(ProbabilityTensor, self).__init__(layer, *args, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.input_spec = [InputSpec(shape=input_shape)]
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis.')

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ProbabilityTensor, self).build()

    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,n 
        #       s.t. \sum_n n = 1
        if isinstance(input_shape, (list,tuple)) and not isinstance(input_shape[0], int):
            input_shape = input_shape[0]

        return (input_shape[0], input_shape[1])

    def squash_mask(self, mask):
        if K.ndim(mask) == 2:
            return mask
        elif K.ndim(mask) == 3:
            return K.any(mask, axis=-1)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        return self.squash_mask(mask)

    def call(self, x, mask=None):
        energy = K.squeeze(self.layer(x), 2)
        p_matrix = K.softmax(energy)
        if mask is not None:
            mask = self.squash_mask(mask)
            p_matrix = make_safe(p_matrix * mask)
            p_matrix = (p_matrix / K.sum(p_matrix, axis=-1, keepdims=True))*mask
        return p_matrix

    def get_config(self):
        config = {}
        base_config = super(ProbabilityTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class SoftAttention(ProbabilityTensor):
    '''This will create the context vector only'''
    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,f where f is weighted features summed across n
        return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask=None):
        if mask is None or mask.ndim==2:
            return None
        else:
            raise Exception("Unexpected situation")

    def call(self, x, mask=None):
        # b,n,f -> b,f via b,n broadcasted
        p_vectors = K.expand_dims(super(SoftAttention, self).call(x, mask), 2)
        expanded_p = K.repeat_elements(p_vectors, K.shape(x)[2], axis=2)
        return K.sum(expanded_p * x, axis=1)


class SoftAttentionConcat(ProbabilityTensor):
    '''This will create the context vector and then concatenate it with the last output of the LSTM'''
    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,f where f is weighted features summed across n
        return (input_shape[0], 2*input_shape[2])

    def compute_mask(self, x, mask=None):
        if mask is None or mask.ndim==2:
            return None
        else:
            raise Exception("Unexpected situation")

    def call(self, x, mask=None):
        # b,n,f -> b,f via b,n broadcasted
        p_vectors = K.expand_dims(super(SoftAttentionConcat, self).call(x, mask), 2)
        expanded_p = K.repeat_elements(p_vectors, K.shape(x)[2], axis=2)
        context = K.sum(expanded_p * x, axis=1)
        last_out = x[:, -1, :]
        return K.concatenate([context, last_out])

class TDistSoftAttention(ProbabilityTensor):
    '''This will create the context vector at each timestep'''
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])

    def compute_mask(self, x, mask=None):
        if mask is None or mask.ndim==2:
            return None
        else:
            raise Exception("Unexpected situation")

    def call(self, x, mask=None):
        # b,n,f -> b,f via b,n broadcasted
        tdcontext = K.zeros_like(x) #context vector that is distributed across the timesteps
        zero_out_time = K.zeros_like(x)
        numtime = max_sen_length
        for t in range(numtime):
            timestep_idx = t+1
            x_in = x
            if timestep_idx < numtime: #when timestep_idx == numtime, we want ALL the timesteps, so no zeroing out in that case
                x_in = K.set_subtensor(x_in[:, timestep_idx:, :], zero_out_time[:, timestep_idx:, :]) #get all LSTM responses up to timestep t, so zero out anything after t
            p_vectors = K.expand_dims(super(TDistSoftAttention, self).call(x_in, mask), 2)
            expanded_p = K.repeat_elements(p_vectors, K.shape(x)[2], axis=2)
            context = K.sum(expanded_p * x_in, axis=1)
            tdcontext = K.set_subtensor(tdcontext[:, t, :], context)
        return tdcontext

class RecurrentMem(Layer):
    '''Abstract base class for recurrent layers that also return their memories.
    Do not use in a model -- it's not a valid layer!
    Use its child class `LSTMMem` instead.
    ```

    # Arguments
        weights: list of numpy arrays to set as initial weights.
            The list should have 3 elements, of shapes:
            `[(input_dim, output_dim), (output_dim, output_dim), (output_dim,)]`.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False). If True, the network will be unrolled,
            else a symbolic loop will be used. When using TensorFlow, the network
            is always unrolled, so this argument does not do anything.
            Unrolling can speed-up a RNN, although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        consume_less: one of "cpu", "mem", or "gpu" (LSTM/GRU only).
            If set to "cpu", the RNN will use
            an implementation that uses fewer, larger matrix products,
            thus running faster on CPU but consuming more memory.

            If set to "mem", the RNN will use more matrix products,
            but smaller ones, thus running slower (may actually be faster on GPU)
            while consuming less memory.

            If set to "gpu" (LSTM/GRU only), the RNN will combine the input gate,
            the forget gate and the output gate into a single matrix,
            enabling more time-efficient parallelization on the GPU. Note: RNN
            dropout must be shared for all gates, resulting in a slightly
            reduced regularization.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        input_length: Length of input sequences, to be specified
            when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
            Note that if the recurrent layer is not the first layer
            in your model, you would need to specify the input length
            at the level of the first layer
            (e.g. via the `input_shape` argument)

    # Input shape
        3D tensor with shape `(nb_samples, timesteps, input_dim)`.

    # Output shape
        - if `return_sequences`: 3D tensor with shape
            `(nb_samples, timesteps, output_dim)`.
        - else, 2D tensor with shape `(nb_samples, output_dim)`.

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.

    # TensorFlow warning
        For the time being, when using the TensorFlow backend,
        the number of timesteps used must be specified in your model.
        Make sure to pass an `input_length` int argument to your
        recurrent layer (if it comes first in your model),
        or to pass a complete `input_shape` argument to the first layer
        in your model otherwise.


    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch.
        This assumes a one-to-one mapping between
        samples in different successive batches.

        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                a `batch_input_shape=(...)` to the first layer in your model.
                This is the expected shape of your inputs *including the batch size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.

        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.

    # Note on using dropout with TensorFlow
        When using the TensorFlow backend, specify a fixed batch size for your model
        following the notes on statefulness RNNs.
    '''
    def __init__(self, weights=None,
                 return_sequences=False, return_mem=True, go_backwards=False, stateful=False,
                 unroll=False, consume_less='cpu',
                 input_dim=None, input_length=None, **kwargs):
        self.return_sequences = return_sequences
        self.return_mem = return_mem
        self.initial_weights = weights
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.consume_less = consume_less

        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RecurrentMem, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.output_dim)
        elif self.return_mem:
            return (input_shape[0], input_shape[1], 2*self.output_dim)
        else:
            return (input_shape[0], self.output_dim)

    def compute_mask(self, input, mask):
        if self.return_sequences or self.return_mem: #this might be questionable
            return mask
        else:
            return None

    def step(self, x, states):
        raise NotImplementedError

    def get_constants(self, x):
        return []

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def preprocess_input(self, x):
        return x

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnnmem(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        elif self.return_mem:
            return K.concatenate(states) #for LSTM, states is the array [h, c] where h is the response c is the memory
        else:
            return last_output

    def get_config(self):
        config = {'return_sequences': self.return_sequences, 'return_mem': self.return_mem,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'consume_less': self.consume_less}
        if self.stateful:
            config['batch_input_shape'] = self.input_spec[0].shape
        else:
            config['input_dim'] = self.input_dim
            config['input_length'] = self.input_length

        base_config = super(RecurrentMem, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTMMem(RecurrentMem):
    '''Long-Short Term Memory unit - Hochreiter 1997.

    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(LSTMMem, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        if self.consume_less == 'gpu':
            self.W = self.init((self.input_dim, 4*self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 4*self.output_dim),
                                     name='{}_U'.format(self.name))

            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           K.get_value(self.forget_bias_init(self.output_dim)),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.U, self.b]
        else:
            self.W_i = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_i'.format(self.name))
            self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_i'.format(self.name))
            self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

            self.W_f = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_f'.format(self.name))
            self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_f'.format(self.name))
            self.b_f = self.forget_bias_init((self.output_dim,),
                                             name='{}_b_f'.format(self.name))

            self.W_c = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_c'.format(self.name))
            self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_c'.format(self.name))
            self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

            self.W_o = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_o'.format(self.name))
            self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_o'.format(self.name))
            self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

            self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                      self.W_c, self.U_c, self.b_c,
                                      self.W_f, self.U_f, self.b_f,
                                      self.W_o, self.U_o, self.b_o]

            self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o])
            self.U = K.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
            self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o])

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            if 0 < self.dropout_W < 1:
                dropout = self.dropout_W
            else:
                dropout = 0
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
                                         input_dim, self.output_dim, timesteps)
            return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
        else:
            return x

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]

        if self.consume_less == 'gpu':
            z = K.dot(x * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) + self.b

            z0 = z[:, :self.output_dim]
            z1 = z[:, self.output_dim: 2 * self.output_dim]
            z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
            z3 = z[:, 3 * self.output_dim:]

            i = self.inner_activation(z0)
            f = self.inner_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.inner_activation(z3)
        else:
            if self.consume_less == 'cpu':
                x_i = x[:, :self.output_dim]
                x_f = x[:, self.output_dim: 2 * self.output_dim]
                x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
                x_o = x[:, 3 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
                x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
                x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
                x_o = K.dot(x * B_W[3], self.W_o) + self.b_o
            else:
                raise Exception('Unknown `consume_less` mode.')

            i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
            f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
            o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

        h = o * self.activation(c)
        return h, [h, c]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(LSTMMem, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AttnFusion(Recurrent):
    '''My own twist to Deep Attention Fusion (http://arxiv.org/pdf/1601.06733v4.pdf), somewhat similar to LSTM.

    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(AttnFusion, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2] #since input is from LSTMMem

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        if self.consume_less == 'gpu':
            self.W = self.init((self.input_dim, 5*self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 4*self.output_dim),
                                     name='{}_U'.format(self.name))

            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           K.get_value(self.forget_bias_init(self.output_dim)),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim), np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.U, self.b]
        else:
            self.W_r = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_r'.format(self.name))
            self.b_r = K.zeros((self.output_dim,), name='{}_b_r'.format(self.name))


            self.W_i = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_i'.format(self.name))
            self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_i'.format(self.name))
            self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

            self.W_f = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_f'.format(self.name))
            self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_f'.format(self.name))
            self.b_f = self.forget_bias_init((self.output_dim,),
                                             name='{}_b_f'.format(self.name))

            self.W_c = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_c'.format(self.name))
            self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_c'.format(self.name))
            self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

            self.W_o = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_o'.format(self.name))
            self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_o'.format(self.name))
            self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

            self.trainable_weights = [self.W_r, self.b_r, self.W_i, self.U_i, self.b_i,
                                      self.W_c, self.U_c, self.b_c,
                                      self.W_f, self.U_f, self.b_f,
                                      self.W_o, self.U_o, self.b_o]

            self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o, self.W_r])
            self.U = K.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
            self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o, self.b_r])

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            if 0 < self.dropout_W < 1:
                dropout = self.dropout_W
            else:
                dropout = 0
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]/2
            timesteps = input_shape[1]
            x_in = x
            cell_in = x
            zeros_mask = K.zeros_like(x)
            x_in = K.set_subtensor(x_in[:, :, input_dim:], zeros_mask[:, :, input_dim:]) #get LSTMMem response (not cell state)
            #x_in = x[:, :, :input_dim] #get LSTMMem response (not cell state)
            #cell_in = x[:, :, input_dim:] #get LSTMMem cell state
            cell_in = K.set_subtensor(cell_in[:, :, :input_dim], zeros_mask[:, :, :input_dim]) #get LSTMMem cell state

            x_i = time_distributed_dense(x_in, self.W_i, self.b_i, dropout,
                                         input_dim*2, self.output_dim, timesteps)
            x_f = time_distributed_dense(x_in, self.W_f, self.b_f, dropout,
                                         input_dim*2, self.output_dim, timesteps)
            x_c = time_distributed_dense(x_in, self.W_c, self.b_c, dropout,
                                         input_dim*2, self.output_dim, timesteps)
            x_o = time_distributed_dense(x_in, self.W_o, self.b_o, dropout,
                                         input_dim*2, self.output_dim, timesteps)
            x_r = time_distributed_dense(x_in, self.W_r, self.b_r, dropout,
                                         input_dim*2, self.output_dim, timesteps)
            return K.concatenate([x_i, x_f, x_c, x_o, x_r, cell_in], axis=2)
        else:
            return x

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]
        input_dim = self.input_dim/2

        if self.consume_less == 'gpu': #x returned was concatenation of response and cell state
            x_in = x
            cell_in = x
            zeros_mask = K.zeros_like(x)
            x_in = K.set_subtensor(x_in[:, input_dim:], zeros_mask[:, input_dim:]) #get LSTMMem response (not cell state)
            cell_in = K.set_subtensor(cell_in[:, :input_dim], zeros_mask[:, :input_dim]) #get LSTMMem cell state
            # x_in = x[:, :dim_in]
            # cell_in = x[:, dim_in:]
            z = K.dot(x_in * B_W[0], self.W[:, :4*self.output_dim]) + K.dot(h_tm1 * B_U[0], self.U[:, :4*self.output_dim]) + self.b[:4*self.output_dim]

            z0 = z[:, :self.output_dim]
            z1 = z[:, self.output_dim: 2 * self.output_dim]
            z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
            z3 = z[:, 3 * self.output_dim: 4*self.output_dim]
            zr = K.dot(x_in * B_W[0], self.W[:, 4*self.output_dim:]) + self.b[4*self.output_dim:] #since recall gate has no U matrix

            i = self.inner_activation(z0)
            f = self.inner_activation(z1)
            r = self.inner_activation(zr)
            c = f * c_tm1 + i * self.activation(z2) + r*cell_in
            o = self.inner_activation(z3)
        else:
            if self.consume_less == 'cpu': #x returned is stack of x_'s and cell_in
                x_i = x[:, :self.output_dim]
                x_f = x[:, self.output_dim: 2 * self.output_dim]
                x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
                x_o = x[:, 3 * self.output_dim: 4*self.output_dim]
                x_r = x[:, 4*self.output_dim: 5*self.output_dim]
                cell_in = x[:, 5*self.output_dim:]
            elif self.consume_less == 'mem': #x returned was concatenation of response and cell state
                x_in = x
                cell_in = x
                zeros_mask = K.zeros_like(x)
                x_in = K.set_subtensor(x_in[:, input_dim:], zeros_mask[:, input_dim:]) #get LSTMMem response (not cell state)
                cell_in = K.set_subtensor(cell_in[:, :input_dim], zeros_mask[:, :input_dim]) #get LSTMMem cell state
                x_i = K.dot(x_in * B_W[0], self.W_i) + self.b_i
                x_f = K.dot(x_in * B_W[1], self.W_f) + self.b_f
                x_c = K.dot(x_in * B_W[2], self.W_c) + self.b_c
                x_o = K.dot(x_in * B_W[3], self.W_o) + self.b_o
                x_r = K.dot(x_in * B_W[4], self.W_r) + self.b_r
            else:
                raise Exception('Unknown `consume_less` mode.')

            i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
            f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
            r = self.inner_activation(x_r)
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c)) + r*cell_in #note cell_in and r need to have same dim
            o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

        h = o * self.activation(c)
        return h, [h, c]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1] #I don't think you need to divide by 2 here since you are preserving the shape
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(5)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(5)])
        return constants

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(AttnFusion, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AttnFusionOnes(Recurrent):
    '''My own twist to Deep Attention Fusion (http://arxiv.org/pdf/1601.06733v4.pdf), somewhat similar to LSTM.
    Zeros mask has been replaced with ones mask

    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(AttnFusionOnes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2] #since input is from LSTMMem

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        if self.consume_less == 'gpu':
            self.W = self.init((self.input_dim, 5*self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 4*self.output_dim),
                                     name='{}_U'.format(self.name))

            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           K.get_value(self.forget_bias_init(self.output_dim)),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim), np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.U, self.b]
        else:
            self.W_r = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_r'.format(self.name))
            self.b_r = K.zeros((self.output_dim,), name='{}_b_r'.format(self.name))


            self.W_i = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_i'.format(self.name))
            self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_i'.format(self.name))
            self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

            self.W_f = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_f'.format(self.name))
            self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_f'.format(self.name))
            self.b_f = self.forget_bias_init((self.output_dim,),
                                             name='{}_b_f'.format(self.name))

            self.W_c = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_c'.format(self.name))
            self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_c'.format(self.name))
            self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

            self.W_o = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_o'.format(self.name))
            self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_o'.format(self.name))
            self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

            self.trainable_weights = [self.W_r, self.b_r, self.W_i, self.U_i, self.b_i,
                                      self.W_c, self.U_c, self.b_c,
                                      self.W_f, self.U_f, self.b_f,
                                      self.W_o, self.U_o, self.b_o]

            self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o, self.W_r])
            self.U = K.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
            self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o, self.b_r])

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            if 0 < self.dropout_W < 1:
                dropout = self.dropout_W
            else:
                dropout = 0
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]/2
            timesteps = input_shape[1]
            x_in = x
            cell_in = x
            zeros_mask = K.ones_like(x)
            x_in = K.set_subtensor(x_in[:, :, input_dim:], zeros_mask[:, :, input_dim:]) #get LSTMMem response (not cell state)
            #x_in = x[:, :, :input_dim] #get LSTMMem response (not cell state)
            #cell_in = x[:, :, input_dim:] #get LSTMMem cell state
            cell_in = K.set_subtensor(cell_in[:, :, :input_dim], zeros_mask[:, :, :input_dim]) #get LSTMMem cell state

            x_i = time_distributed_dense(x_in, self.W_i, self.b_i, dropout,
                                         input_dim*2, self.output_dim, timesteps)
            x_f = time_distributed_dense(x_in, self.W_f, self.b_f, dropout,
                                         input_dim*2, self.output_dim, timesteps)
            x_c = time_distributed_dense(x_in, self.W_c, self.b_c, dropout,
                                         input_dim*2, self.output_dim, timesteps)
            x_o = time_distributed_dense(x_in, self.W_o, self.b_o, dropout,
                                         input_dim*2, self.output_dim, timesteps)
            x_r = time_distributed_dense(x_in, self.W_r, self.b_r, dropout,
                                         input_dim*2, self.output_dim, timesteps)
            return K.concatenate([x_i, x_f, x_c, x_o, x_r, cell_in], axis=2)
        else:
            return x

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]
        input_dim = self.input_dim/2

        if self.consume_less == 'gpu': #x returned was concatenation of response and cell state
            x_in = x
            cell_in = x
            zeros_mask = K.ones_like(x)
            x_in = K.set_subtensor(x_in[:, input_dim:], zeros_mask[:, input_dim:]) #get LSTMMem response (not cell state)
            cell_in = K.set_subtensor(cell_in[:, :input_dim], zeros_mask[:, :input_dim]) #get LSTMMem cell state
            # x_in = x[:, :dim_in]
            # cell_in = x[:, dim_in:]
            z = K.dot(x_in * B_W[0], self.W[:, :4*self.output_dim]) + K.dot(h_tm1 * B_U[0], self.U[:, :4*self.output_dim]) + self.b[:4*self.output_dim]

            z0 = z[:, :self.output_dim]
            z1 = z[:, self.output_dim: 2 * self.output_dim]
            z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
            z3 = z[:, 3 * self.output_dim: 4*self.output_dim]
            zr = K.dot(x_in * B_W[0], self.W[:, 4*self.output_dim:]) + self.b[4*self.output_dim:] #since recall gate has no U matrix

            i = self.inner_activation(z0)
            f = self.inner_activation(z1)
            r = self.inner_activation(zr)
            c = f * c_tm1 + i * self.activation(z2) + r*cell_in
            o = self.inner_activation(z3)
        else:
            if self.consume_less == 'cpu': #x returned is stack of x_'s and cell_in
                x_i = x[:, :self.output_dim]
                x_f = x[:, self.output_dim: 2 * self.output_dim]
                x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
                x_o = x[:, 3 * self.output_dim: 4*self.output_dim]
                x_r = x[:, 4*self.output_dim: 5*self.output_dim]
                cell_in = x[:, 5*self.output_dim:]
            elif self.consume_less == 'mem': #x returned was concatenation of response and cell state
                x_in = x
                cell_in = x
                zeros_mask = K.ones_like(x)
                x_in = K.set_subtensor(x_in[:, input_dim:], zeros_mask[:, input_dim:]) #get LSTMMem response (not cell state)
                cell_in = K.set_subtensor(cell_in[:, :input_dim], zeros_mask[:, :input_dim]) #get LSTMMem cell state
                x_i = K.dot(x_in * B_W[0], self.W_i) + self.b_i
                x_f = K.dot(x_in * B_W[1], self.W_f) + self.b_f
                x_c = K.dot(x_in * B_W[2], self.W_c) + self.b_c
                x_o = K.dot(x_in * B_W[3], self.W_o) + self.b_o
                x_r = K.dot(x_in * B_W[4], self.W_r) + self.b_r
            else:
                raise Exception('Unknown `consume_less` mode.')

            i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
            f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
            r = self.inner_activation(x_r)
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c)) + r*cell_in #note cell_in and r need to have same dim
            o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

        h = o * self.activation(c)
        return h, [h, c]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1] #I don't think you need to divide by 2 here since you are preserving the shape
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(5)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(5)])
        return constants

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(AttnFusionOnes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))