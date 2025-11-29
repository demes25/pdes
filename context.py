import tensorflow as tf

PACKAGE : str = 'Lagrangian'

# define global things here
DTYPE : tf.DType = tf.float32

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
