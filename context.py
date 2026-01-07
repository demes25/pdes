import tensorflow as tf
from typing import Sequence
from keras import backend
PACKAGE : str = 'PDEs'

MIXED = False

# define global things here
DTYPE : tf.DType = tf.float32
backend.set_floatx(DTYPE.name)

if MIXED:
    from keras import mixed_precision

    mixed_precision.set_global_policy('mixed_float16')


Shape = Sequence[int] | tf.TensorShape | tf.Tensor

# some useful c-numbers
zero : tf.Tensor = tf.constant(0, dtype=DTYPE)
one : tf.Tensor = tf.constant(1, dtype=DTYPE)
two : tf.Tensor = tf.constant(2.0, dtype=DTYPE)
pi : tf.Tensor = tf.constant(3.14159265359, dtype=DTYPE)
half : tf.Tensor = tf.constant(0.5, dtype=DTYPE)
quarter : tf.Tensor = tf.constant(0.25, dtype=DTYPE)

deriv_step : tf.Tensor = tf.constant(1e-3, dtype=DTYPE)

# PHYSICAL CONSTANTS
# I have set these to natural units
G : tf.Tensor = one 
c : tf.Tensor = one 
h : tf.Tensor = one
