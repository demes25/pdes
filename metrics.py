# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Statistics

from context import *
from lattices import Image
# some statistical functions to test the performance of our models.

# finds the r-squared value given the forcing term and the reconstructed forcing term
# returns a scalar tensor.
def RR(force : Image, reconstruced_force : Image, epsilon=zero) -> tf.Tensor:
    force_grid = force.view()
    reconstruced_force_grid = reconstruced_force.view()

    flat_force = tf.reshape(force_grid, shape=[-1])
    flat_rec_force = tf.reshape(reconstruced_force_grid, shape=[-1])

    mean_flat_force = tf.reduce_mean(flat_force, axis=0, keepdims=True)

    squares = tf.reduce_sum(tf.square(mean_flat_force - flat_force), axis=0)
    res_squares = tf.reduce_sum(tf.square(flat_rec_force - flat_force), axis=0)

    return one - res_squares/(squares + epsilon)


    