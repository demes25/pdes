# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Fields

# this is a quick demonstration of a neural network that
# offers some learned solution to a given lagrangian density over 
# a given domain.

import tensorflow as tf
from keras import models, layers, utils
from typing import List
from context import DTYPE, PACKAGE

from operators import Image, Operator


# a general parent class for our networks
# this simply establishes the notion that the solution must exist in some n-dimensional manifold ('dims')
# and have some consistent output shape for all input. 
@utils.register_keras_serializable(package=PACKAGE)
class SolutionNetwork(models.Model):
    def __init__(
        self, 
        dims, # amount of dimensions
        shape : List[int] = [], # output shape - by default a scalar field
        
        dtype = DTYPE, # we choose dtype, if wanted
        **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)

        self.dims = dims # amount of spatial dimensions
        self.shape = shape 

    # we add the standard boilerplate for serializability
    @classmethod 
    def from_config(cls, config : dict):
        return cls(**config)
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'dims' : self.dims,
            'shape' : self.shape
        })
        return config
    

    # a function to act an instance of the inheriting class on an Image object to be called on by an instance of the inheriting class.
    # by default, it returns the input image
    def act_on_image(self, image : Image) -> Image:
        return image


# wraps our solution networks as operators.
def NetworkOperator(U : SolutionNetwork) -> Operator:
    return U.act_on_image



# --- THE SPATIAL KERNEL --- #

# this is a multi-kernel convolutional layer that works on
# a given amount of spatial dimensions
#
# TODO: work on time-dependence and causality.
@utils.register_keras_serializable(package=PACKAGE)
class KernelLayer(layers.Layer):
    def __init__(self, 
                dims = 1, # amount of spatial dimensions
                filters = 64, # output channels 
                size=3, # kernel size
                strides=1, # standard convolution arguments
                padding='same',
                activation='gelu',
                
                transpose=False, # whether the kernel should be transposed (i.e. go the 'opposite' direction)
                pool_size=None, # whether to pool at the end

                dtype=DTYPE,
                **kwargs
        ):

        # TODO: generalize
        assert 0 < dims <= 3, 'Unable to handle less than one or more than three spatial dimensions.'
        
        super().__init__(dtype=dtype, **kwargs)
        
        # for serializability
        self.internal_config = {
            'dims' : dims,
            'filters' : filters,
            'size' : size,
            'strides' : strides,
            'padding' : padding,
            'activation' : activation,

            'transpose' : transpose,
            'pool_size' : pool_size
        }

        # our choice of convolution depends on our amount of spatial dimensions
        if transpose:
            conv = tuple([layers.Conv1DTranspose, layers.Conv2DTranspose, layers.Conv3DTranspose])[dims - 1]
        else:
            conv = tuple([layers.Conv1D, layers.Conv2D, layers.Conv3D])[dims - 1] 
        
        self.conv = conv(filters=filters, kernel_size=size, strides=strides, padding=padding, activation=activation, dtype=dtype)
        
        if pool_size is not None:
            pool_type = tuple([layers.MaxPool1D, layers.MaxPool2D, layers.MaxPool3D])[dims-1] # again take into account spatial dimension count
            self.pooling = pool_type(pool_size=pool_size, padding=padding)

        else:
            self.pooling = None 

    # we add the standard boilerplate for serializability
    @classmethod
    def from_config(cls, config : dict) -> 'KernelLayer':
        return cls(**config)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(self.internal_config)
        return config
    

    # tensor - an input of shape [B, n1, n2, ..., nN, I]
    # where N is the dimension count, and I is the input channel count
    #
    # returns convolved+pooled tensor of shape [B, m1, m2, ..., mN, O]
    # where m... depend on convolution arguments, and O is the output channel count.
    def call(self, tensor):
        conv = self.conv(tensor)
        return self.pooling(conv) if self.pooling is not None else conv


# This is built based on the U-Net architecture
# https://www.sciencedirect.com/topics/computer-science/u-net
#
# NOTE: This system currently only supports square (or cube) lattices with side length divisible by pool_size^depth.
# i.e. a tensor of shape [B, (T), N, N, N, *S].
# note that N here must take into account the relevant padding. so for a square mesh with unpadded side length L,
# (L + 2*padding) must be divisible by (pool_size^depth)
#
# TODO: generalize from the above restraint
#
# TODO: ENFORCE NAN CHECKS!
@utils.register_keras_serializable(package=PACKAGE)
class SpatialKernel(SolutionNetwork):
    
    def __init__(self, 
                 dims, # amount of SPATIAL dimensions
                 shape : List[int] = [], # output shape - by default a scalar field
                 
                 size = 3, # kernel size

                 depth = 5, # encoder depth, should be a function of the size of our desired domain and the pooling factor.
                 # for the receptive field to cover the entire domain lattice, the depth and pool size should be selected such that
                 # pool_size ^ depth >= lattice side length

                 pool_size = 2, # the pooling factor. # TODO: make this general along each spatial dimension
                 
                 init_filters = 16, # the amount of filters that the first kernel has. this will scale by *pool_size per each downsizing.
                 activation = 'gelu',
                 
                 dtype = DTYPE, # we choose dtype, if wanted
                 **kwargs
                 ):

        super().__init__(dims=dims, shape=shape, dtype=dtype, **kwargs)

        # for serializability
        self.internal_config = {
            'size' : size,
            'depth' : depth,
            'init_filters' : init_filters,
            'pool_size' : pool_size,
            'activation' : activation
        }

        # 'output channels' will encode our desired output shape.
        output_channels = 1
        for s in shape:
            output_channels *= s

        self.shape = shape
        self._dtype = dtype
        self.dims = dims
        self.depth = depth
        
        # the first layer -- encoder
        # 'encodes' the forcing signal
        # consists of a chain of convolutions and pooling, where 
        # each time we increase the filter count.
        self.encoders = [] # the convolutions
        self.downsizers = [] # the pooling (really these are pooled convolutions, not pure pooling)
        
        k = init_filters

        for _ in range(depth):
            enc = KernelLayer(dims=dims, filters=k, size=size, padding='same', activation=activation)
            k *= pool_size
            down = KernelLayer(dims=dims, filters=k, size=size, padding='same', activation=activation, pool_size=pool_size)

            self.encoders.append(enc)
            self.downsizers.append(down)
        
        # the second layer -- bottleneck
        # consists of a couple high-filter convolutions.
        self.bottleneck1 = KernelLayer(dims=dims, filters=k, size=size, padding='same', activation=activation)
        self.bottleneck2 = KernelLayer(dims=dims, filters=k, size=size, padding='same', activation=activation)

        # the third layer -- decoder
        # consists of some strided convolutions, concatenations and transpose convolutions that amplify the solution signal back.
        self.upsizers = [] # the strided transpose convolutions
        self.decoders = [] # the unstrided transpose convolutions

        for _ in range(depth):
            k //= pool_size
            up = KernelLayer(dims=dims, filters=k, size=size, padding='same', activation=activation, transpose=True, strides=pool_size)
            dec = KernelLayer(dims=dims, filters=k, size=size, padding='same', activation=activation)

            self.upsizers.append(up)
            self.decoders.append(dec)

        # the final layer - output
        self.output_layer = KernelLayer(dims=dims, filters=output_channels, size=size, activation=None)


    # we add the standard boilerplate for serializability
    @classmethod 
    def from_config(cls, config : dict):
        return cls(**config)
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update(self.internal_config)
        return config
    

    # input : domain [B, X, Y, Z, ..., -1] - a lattice with its entry_shape squished to [-1]
    # output : [B, X', Y', Z', ..., *self.shape] - a set of B tensors corresponding to the field values at each given point 
    def call(self, x : tf.Tensor) -> tf.Tensor:

        # encode + track encodings + downsize
        encodings = [] # tf.TensorArray(dtype=self._dtype, size=self.depth, dynamic_size=False, clear_after_read=True)
        y = x # the starting y is our initial input
        for enc, down in zip(self.encoders, self.downsizers):
            # repeatedly encode and downsize, store the encodings to later append to corresponding decodings
            y = enc(y) # encode -> now y holds the encoding of whatever y previously was
            encodings.append(y) # append the encodings array before we reassign 
            y = down(y) # reassign to y the downsizing of the encoding

        # bottleneck
        y = self.bottleneck1(y)
        y = self.bottleneck2(y)

        # upsize + concatenate earlier encodings + decode
        for up, dec in zip(self.upsizers, self.decoders):
            y = up(y)
            # concatenate to the corresponding encoding. this will effectively follow the encodings array in reverse, 
            # so we can proceed by repeatedly popping from the end
            y0 = encodings.pop(-1)
            y = tf.concat([y, y0], axis=-1) # concatenate the two and pass to the decoder
            y = dec(y)

        # output
        out = self.output_layer(y) # pass through the output layer
        
        return out
        

    # we act on an image
    # since the image holds [B, (T), X, Y, Z, ..., *entry_shape], and call() expects:
    #       - no temporal dimension
    #       - the value shape to be flattened at the end,
    #
    # we must first reshape accordingly before we can do call
    def act_on_image(self, image : Image) -> Image:
        # we check if the image contains a temporal dimension.
        
        leading_shape = image.leading_shape

        # if our domain is 'timed':
        if image.domain.timed:
            start_shape = [image.batches * leading_shape[1]] # the first two axes are batch and time
            mid_shape = leading_shape[2:] 
            end_shape = [-1]

            input_shape = tf.concat([start_shape, mid_shape, end_shape], axis=0)
        
        else:
            # otherwise: the first axis is B and the rest of the lattice is purely spatial
            input_shape = tf.concat([leading_shape, [-1]], axis=0)

        x = tf.reshape(image.view(padded=True), shape=input_shape)

        # we preparet to reshape the output to [B, (T), X, Y, Z, ..., *self.shape] 
        output_shape = tf.concat([leading_shape, self.shape], axis=0)
        
        # we act on the properly shaped input x, and cast to the desired output shape
        output = tf.reshape(self(x), shape=output_shape)

        # return the resulting image
        return Image(domain=image.domain, grid=output, shape=output_shape, entry_shape=self.shape)