# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Fields

# this is a quick demonstration of a neural network that
# offers some learned solution to a given lagrangian density over 
# a given domain.

import tensorflow as tf
from keras import models, layers, utils
from geometry import N, DTYPE

# wraps a simple Sequential model
# 
# we apply a neural network to some discretization of our desired domain, essentially
# finite element method, pass everything through a dense, and train on minimizing the total action.
#
# we make these spacetime fields - each point in the domain has four coordinates.
@utils.register_keras_serializable
class Field(models.Model):
    
    def __init__(self, 
                 rank = 0, # by default a scalar field
                 hidden_dims = 64, # the hidden dimension of our mlp layers 
                 activation = 'gelu', # activation - i will put gelu for balance between universal differentiability and neuron expressivity - sigmoids suppress large numbers
                 dtype = DTYPE # we choose dtype, if wanted
                 ):
        super().__init__()

        # for serializability
        self.internal_config = {
            'rank' : rank,
            'hidden_dims' : hidden_dims,
            'activation' : activation,
            'dtype' : dtype
        }

        self.rank = rank 
        self.output_shape = [] if rank == 0 else rank * [N]
        self._dtype = dtype 

        # we will make a three-layer mlp
        self.fxn = models.Sequential(
            layers= [
                layers.Dense(hidden_dims, activation=activation, dtype=dtype), 
                layers.Dense(hidden_dims, activation=activation, dtype=dtype),
                layers.Dense(hidden_dims, activation=activation, dtype=dtype), # note that gelu suppresses negative numbers
                layers.Dense(rank * N, dtype=dtype) # so we put no activation for last layer, we want to be able to have negative numbers
            ]
        )

    # we add the standard boilerplate for serializability
    @classmethod
    def from_config(cls, config : dict):
        return cls(**config)

    def get_config(self):
        return {**self.internal_config, **super().get_config()}
    
    # again boilerplate for building
    def build(self, input_shape):
        dummy_input = tf.zeros(shape=input_shape, dtype=self._dtype)
        _ = self.fxn(dummy_input)
        super().build(input_shape)


    # input : domain [N, 4] - a set of N discrete 4-coordinate events in spacetime
    # output : [N, rank * [4]] - a set of N tensors corresponding to the field values at each given point 
    def call(self, domain):
        result = self.fxn(domain)
        result = tf.reshape(result, self.output_shape)
        return result 
        
    
