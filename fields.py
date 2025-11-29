# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Fields

# this is a quick demonstration of a neural network that
# offers some learned solution to a given lagrangian density over 
# a given domain.

import tensorflow as tf
from keras import models, layers, utils
from context import DTYPE, PACKAGE


# this is a multi-kernel convolutional layer that works on
# the given amount of spatial dimensions
#
# TODO: work on time-dependence and causality.
@utils.register_keras_serializable(package=PACKAGE)
class SpatialKernel(layers.Layer):
    # TODO: MAKE
    def __init__(self, 
                dims = 1, # amount of spatial dimensions
                first_channels = 16, # output channels for the first kernel
                second_channels = 32, # output channels for the second kernel
                size=3, # kernel size
                strides=1, # standard convolution arguments
                padding='same',
                activation='gelu',
                
                transpose=False, # whether the kernel should be transposed (i.e. go the 'opposite' direction)
                pool_size=None, # whether to pool at the end

                dtype=DTYPE,
                **kwargs
        ):
        
        dtype = tf.as_dtype(dtype)

        # TODO: generalize
        assert dims < 3, 'Unable to handle less than one or more than three spatial dimensions.'
        
        super().__init__(dtype=dtype, **kwargs)
        
        # for serializability
        self.internal_config = {
            'dims' : dims,
            'first_channels' : first_channels,
            'second_channels' : second_channels,
            'size' : size,
            'strides' : strides,
            'padding' : padding,
            'activation' : activation,

            'transpose' : transpose,
            'pool_size' : pool_size,

            'dtype' : dtype.name
        }

        # our choice of convolution depends on our amount of spatial dimensions
        if transpose:
            conv = tuple(layers.Conv1DTranspose, layers.Conv2DTranspose, layers.Conv3DTranspose)[dims - 1]
        else:
            conv = tuple(layers.Conv1D, layers.Conv2D, layers.Conv3D)[dims - 1] 

        self.first_conv = conv(filters=first_channels, kernel_size=size, strides=strides, padding=padding, activation=activation, dtype=dtype)
        self.second_conv = conv(filters=second_channels, kernel_size=size, strides=strides, padding=padding, activation=activation, dtype=dtype)    
        
        if pool_size is not None:
            pool_type = tuple(layers.MaxPool1D, layers.MaxPool2D, layers.MaxPool3D)[dims-1] # again take into account spatial dimension count
            self.pooling = pool_type(pool_size=pool_size, padding=padding)

        else:
            self.pooling = None 

    # we add the standard boilerplate for serializability
    @classmethod
    def from_config(cls, config : dict) -> 'SpatialKernel':
        return cls(**config)

    def get_config(self) -> dict:
        return {**self.internal_config, **super().get_config()}
    

    # tensor - an input of shape 
    def call(self, tensor):
        first_layer = self.first_conv(tensor)
        second_layer = self.second_conv(first_layer)
        if self.pooling is not None:
            return self.pooling(second_layer)
        return second_layer


# This is built based on the U-Net architecture
# https://www.sciencedirect.com/topics/computer-science/u-net
# TODO: FINISH THIS
@utils.register_keras_serializable(package=PACKAGE)
class SpatialField(models.Model):
    
   def __init__(self, 
                 shape = [], # output shape - by default a scalar field
                 
                 dims = 1, # amount of spatial dimensions
                 size = 3, # kernel size
                 padding='same',
                 activation='gelu',
                
                 dtype = DTYPE, # we choose dtype, if wanted
                 **kwargs
                 ):
        
        dtype = tf.as_dtype(dtype)

        super().__init__(dtype=dtype, **kwargs)

        # for serializability
        self.internal_config = {
            'shape' : shape,
            'dims' : dims,
            'size' : size,
            'padding' : padding,
            'activation' : activation,
            'dtype' : dtype.name
        }

        self.shape = shape
        self._dtype = dtype
        
        # we will make a three-layer mlp
        self.encoder = models.Sequential(
            layers= [
                layers.Dense(hidden_dims, activation=activation, dtype=dtype),
                layers.Dense(hidden_dims, activation=activation, dtype=dtype),
                layers.Dense(hidden_dims, activation=activation, dtype=dtype), # note that gelu suppresses negative numbers
                layers.Dense(flat_shape, dtype=dtype) # so we put no activation for last layer, we want to be able to have negative numbers
            ]
        )

    # we add the standard boilerplate for serializability
    @classmethod
    def from_config(cls, config : dict) -> 'SpatialField':
        return cls(**config)

    def get_config(self) -> dict:
        return {**self.internal_config, **super().get_config()}
    
    # again boilerplate for building
    def build(self, input_shape):
        dummy_input = tf.zeros(shape=input_shape, dtype=self._dtype)
        _ = self.fxn(dummy_input)
        super().build(input_shape)


    # input : domain [B, X, Y, Z, ...] - a set of B discrete n-coordinate events in spacetime
    # output : [B, S] - a set of B tensors corresponding to the field values at each given point 
    def call(self, domain : tf.Tensor) -> tf.Tensor:
        B = tf.shape(domain)[0]
        B = tf.expand_dims(B, axis=0)

        result = self.fxn(domain)
        result = tf.reshape(result, tf.concat([B, self.shape], axis=0))
        return result 
        
    
