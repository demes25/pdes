# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Operators

import tensorflow as tf
from params import *

# here we define algebra for functional addition, multiplication, more general operators
# also we define gradients via tf.GradientTape

# some nice constants
zero = tf.constant(0, dtype=DTYPE)
one = tf.constant(1, dtype=DTYPE)
half = tf.constant(0.5, dtype=DTYPE)


# we define the constant function - if necessary
def constant_fn(c):
    def _h(x):
        return c
    return _h

# for all these, *fns is assumed to be a bunch of functions/functionals

def scale_fn(c, F):
    def _h(x):
        return c * F(x)
    return _h

def add_fn(*fns):
    def _h(x):
        return tf.add_n([f(x) for f in fns])
    return _h
    
def mul_fn(*fns):
    def _h(x):
        # no such 'mul_n' so we multiply manually
        result = fns[0](x)
        for i in range(1, len(fns)):
            result = result * fns[i](x)
        return result
    return _h

def apply_fn(fn, *fns, **kwargs):
    # we assume fns contains ONLY FUNCTIONS
    # and we apply 'fn' to them at some point x.
    # any other arguments to fn may be included in kwargs
    def _h(x):
        eval_fns = [f(x) for f in fns]
        return fn(*eval_fns, **kwargs)

    return _h 


# we define some operators as well

# F a rank n tensor
# returns F' a rank n+1 tensor
def grad(F):
    def dF(x):
        with tf.GradientTape() as tape:
            # we watch our inputs
            tape.watch(x)

            # returns rank n+1 tensor F', where F'[..., mu] = partial_mu F  
            f = F(x)
        return tape.gradient(f, x)
    
    return dF
