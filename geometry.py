# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Geometry

import tensorflow as tf

# Here we will define some geometric notions such as metrics and covariant derivatives

# set the amount of components to 4 (spacetime)
N = 4

# we define general algebra for callable addition, multiplication, and more general operations.
def add_fn(*args):
    def _h(x):
        return tf.add_n([f(x) for f in args])
    return _h
    
def mul_fn(*args):
    def _h(x):
        # no such 'mul_n' so we multiply manually
        result = args[0](x)
        for i in range(1, len(args)):
            result = result * args[i](x)
        return result
    return _h

def apply_fn(fn, *args, **kwargs):
    # we assume args contains ONLY FUNCTIONS
    # and we apply 'fn' to them at some point x.
    # any other arguments to fn may be included in kwargs
    def _h(x):
        eval_args = [f(x) for f in args]
        return fn(*eval_args, **kwargs)

    return _h 


# we define some general operators as well
def grad(*)
