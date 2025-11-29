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

from operators import Image


# this is a multi-kernel convolutional layer that works on
# the given amount of spatial dimensions
#
# TODO: work on time-dependence and causality.
@utils.register_keras_serializable(package=PACKAGE)
class KernelLayer(layers.Layer):
    # TODO: MAKE
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
        return {**self.internal_config, **super().get_config()}
    

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
@utils.register_keras_serializable(package=PACKAGE)
class SpatialKernel(models.Model):
    
    def __init__(self, 
                 shape : List[int] = [], # output shape - by default a scalar field
                 
                 dims = 1, # amount of spatial dimensions
                 size = 3, # kernel size

                 activation = 'gelu',
                
                 dtype = DTYPE, # we choose dtype, if wanted
                 **kwargs
                 ):

        super().__init__(dtype=dtype, **kwargs)

        # for serializability
        self.internal_config = {
            'shape' : shape,
            'dims' : dims,
            'size' : size,
            'activation' : activation
        }

        # 'output channels' will encode our desired output shape.
        output_channels = 1
        for s in shape:
            output_channels *= s

        self.shape = shape
        self._dtype = dtype
        self.dims = dims
        
        # the first layer -- encoder
        # 'encodes' the forcing signal
        # consists of a chain of convolutions and pooling, where 
        # each time we increase the filter count.
        self.enc1 = KernelLayer(dims=dims, filters=64, size=size, padding='same', activation=activation)
        self.down1 = KernelLayer(dims=dims, filters=128, size=size, padding='same', activation=activation, pool_size=2)
        
        self.enc2 = KernelLayer(dims=dims, filters=128, size=size, padding='same', activation=activation)
        self.down2 = KernelLayer(dims=dims, filters=256, size=size, padding='same', activation=activation, pool_size=2)

        self.enc3 = KernelLayer(dims=dims, filters=256, size=size, padding='same', activation=activation)
        self.down3 = KernelLayer(dims=dims, filters=512, size=size, padding='same', activation=activation, pool_size=2)

        
        # the second layer -- bottleneck
        # consists of a couple high-filter convolutions.
        self.bottleneck1 = KernelLayer(dims=dims, filters=512, size=size, padding='same', activation=activation)
        self.bottleneck2 = KernelLayer(dims=dims, filters=512, size=size, padding='same', activation=activation)

        # the third layer -- decoder
        # consists of some concatenations and transpose convolutions that amplify the solution signal back.
        self.up1 = KernelLayer(dims=dims, filters=256, size=size, padding='same', activation=activation, transpose=True, strides=2)
        self.dec1 = KernelLayer(dims=dims, filters=256, size=size, padding='same', activation=activation)

        self.up2 = KernelLayer(dims=dims, filters=128, size=size, padding='same', activation=activation, transpose=True, strides=2)
        self.dec2 = KernelLayer(dims=dims, filters=128, size=size, padding='same', activation=activation)

        self.up3 = KernelLayer(dims=dims, filters=64, size=size, padding='same', activation=activation, transpose=True, strides=2)
        self.dec3 = KernelLayer(dims=dims, filters=64, size=size, padding='same', activation=activation)

        # the final layer - output
        self.output_layer = KernelLayer(dims=dims, filters=output_channels, size=size, activation=None)
        

    # we add the standard boilerplate for serializability
    @classmethod 
    def from_config(cls, config : dict):
        return cls(**config)
    
    def get_config(self) -> dict:
        return {**self.internal_config, **super().get_config()}
    

    # input : domain [B, X, Y, Z, ...] - a mesh
    # output : [B, X', Y', Z', ..., self.shape] - a set of B tensors corresponding to the field values at each given point 
    def call(self, x : tf.Tensor) -> tf.Tensor:
        # encode
        e1 = self.enc1(x)   
        d1 = self.down1(e1)    

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        d3 = self.down3(e3)

        # bottleneck
        b = self.bottleneck1(d3)
        b = self.bottleneck2(b)

        # decode + concatenate earlier encodings
        u1 = self.up1(b)
        u1 = tf.concat([u1, e3], axis=-1)
        u1 = self.dec1(u1)

        u2 = self.up2(u1)
        u2 = tf.concat([u2, e2], axis=-1)
        u2 = self.dec2(u2)

        u3 = self.up3(u2)
        u3 = tf.concat([u3, e1], axis=-1)
        u3 = self.dec3(u3)

        # output
        out = self.output_layer(u3)
        out = tf.reshape(out, tf.concat([tf.shape(out)[:-1], self.shape], axis=0))

        return out
        


# wraps models as operators.
def OperatorWrapper(U : models.Model):
    def _h(phi : Image) -> Image:
        
        re_shape = tf.concat([phi.padded_mesh_shape, [-1]], axis=0)
        x = tf.expand_dims(tf.reshape(phi.mesh, re_shape), axis=0)

        Uphi = U(x)
        Uphi = tf.squeeze(Uphi, axis=0)

        return Image(phi.domain, mesh=Uphi, geometry=phi.geometry, pad=phi.pad)

    return _h




