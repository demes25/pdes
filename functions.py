# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Function Composition

import tensorflow as tf
from context import *
from typing import Callable

# here we define what a function is.
# 
# we assume that functions will act individually on points
# and therefore do not need to decode mesh structure like operators do.
# 
# we will assume that the input tensor for any function look slike [B, ...]
# where the mesh axes are flattened into the batch dimension.

# TODO: wrap Functions in a bona-fide class so we can know things like input/output dimensions, etc.

# here we define algebra for functional addition, multiplication, more general operators
# also we define gradients via tf.GradientTape
Function = Callable[..., tf.Tensor]
Functional = Callable[..., Function]

# some functions and function generators

# metric must be [B, N, N] or None
def norm_sq_fn(metric : tf.Tensor | None = None) -> Function:

    if metric is None:
        # returns [B]
        return apply_fn(tf.reduce_sum, tf.square, axis=-1)
    
    # x is [B, N]
    def _h(x):
        # returns [B]
        return tf.einsum('bi,bij,bj->b', x, metric, x)
    return _h 

# mu must be [N]
# var must be scalar
def gaussian_fn(mu : tf.Tensor, var : tf.Tensor) -> Function:
    var2 = two * var
    factor = tf.sqrt(pi * var2)

    # mu must be some dimension [N]
    # now we make it [1, N] to fit with...
    mu = tf.expand_dims(mu, axis=0)

    # x, which must be [B, N]
    def _h(x : tf.Tensor) -> tf.Tensor:
        k = tf.reduce_sum(tf.square(x - mu), axis=-1)/var2

        # the result is [B]
        return tf.exp(-k)/factor

    return _h 


# we define the constant function - if necessary
def constant_fn(c : tf.Tensor) -> Function:
    def _h(x) -> tf.Tensor:
        return c
    return _h

# --- COMPOSITIONS --- #

# We define some composition functionals
# for all these, *fnargs is assumed to be a bunch of functions/functionals

def add_fn(*fnargs : Function) -> Function:
    def _h(x) -> tf.Tensor:
        return tf.add_n([f(x) for f in fnargs])
    return _h
    
def mul_fn(*fnargs : Function) -> Function:
    def _h(x) -> tf.Tensor:
        # no such 'mul_n' so we multiply manually
        result = fnargs[0](x)
        for i in range(1, len(fnargs)):
            result = result * fnargs[i](x)
        return result
    return _h

def apply_fn(fn : Callable, *fnargs : Function, **kwargs) -> Function:
    # we assume fnargs contains ONLY FUNCTIONS
    # and we apply 'fn' to them at some point x.
    # any other arguments to fn may be included in kwargs
    def _h(x) -> tf.Tensor:
        eval_fnargs = [f(x) for f in fnargs]
        return fn(*eval_fnargs, **kwargs)

    return _h 


def scale_fn(c : tf.Tensor, f : Function) -> Function:
    def _h(x) -> tf.Tensor:
        return c * f(x)
    return _h

def sub_fn(f : Function, g : Function) -> Function:
    def _h(x) -> tf.Tensor:
        return f(x) - g(x)
    return _h 

def div_fn(f : Function, g : Function) -> Function:
    def _h(x) -> tf.Tensor:
        return f(x)/g(x)
    return _h


# einstein summation functional
def einsum_fn(instructions : str, *fnargs : Function):
    def _h(x) -> tf.Tensor:
        eval_fnargs = [f(x) for f in fnargs]
        return tf.einsum(instructions, *eval_fnargs)
    return _h 
    
    
# G a rank 2 tensor function
def det(G : Function) -> Function:
    def _h(x) -> tf.Tensor:
        return tf.linalg.det(G(x))
    
    return _h



