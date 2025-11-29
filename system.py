# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Systems
# TODO

import tensorflow as tf
from keras.optimizers import Optimizer
from keras.models import Model
from keras.utils import Progbar

from functions import *
from geometry import Geometry
from lagrangians import Functional, FunctionalGenerator
from algebra import Norm

# here we look at general systems and how to deal with them:
# discretizing the domain, passing through neural networks, 
# applying variations and boundary conditions

# a helper function to form convex combinations with a Bernoulli parameter p
# note this is diff'able in A and B, not in p.
def convex_combo(A : tf.Tensor, B : tf.Tensor, p : tf.Tensor) -> tf.Tensor:
    if p == zero:
        return A
    elif p == one:
        return B
    else:
        return (one - p) * A + p * B 


class System:
    def __init__(
        self,
        parameter_num, # how many parameters do we have? (4 for fields, 1 (time) for regular)

        geometry : Geometry,  # (g, g^-1)
        # the metrics for our underlying geometry. we will use this to generate the variation
        # and calculate the boundary penalty
        # defaults to euclidean if None
    
        # TODO: mak
        boundary_function : Function, # function that gives us the desired boundary values

        variation_generator : FunctionalGenerator, 
        # a variation generator from the 'variations' module.
        # note that this thing is a function that takes in a field/soltuion A
        # and returns a function L[A] which acts on coordinates

        *variation_args, # args for the variation generator
        **variation_kwargs, # kwargs for the variation generator
    ):
        self.parameter_num = parameter_num
        self.variation = variation_generator(*variation_args, **variation_kwargs, geometry=geometry)
        self.boundary_function = boundary_function
        self.metric, self.inv_metric = geometry

    # I will assume that all proposed solutions are either scalar
    # or covariant, and therefore I will be taking norms using the inverse metric 
    # (as opposed to the regular metric, as one would for contravariant objects)


    # we calculate the action for some 
    # proposed solution U
    #
    # this returns a function dS : [B, N] -> []
    # which takes in a discretized domain and returns the action of U over that domain
    #
    # note that all of our variation generators already include the volume form
    #
    # this is really the total action variation over the domain
    def action(self, U : Function):
        a = apply_fn(tf.reduce_sum, self.variation(U), axis=0)

        return a
    

    # we calculate the boundary penalty for some
    # proposed solution U
    #
    # this returns a function P : [B, N] -> []
    # which takes in a discretized boundary and returns the mean square difference between U and our
    # boundary function over the given points.
    def boundary_penalty(self, U : Function):
        # we first take differences
        dif = sub_fn(U, self.boundary_function)
        
        # then we take norms (Norm remains squared, recall)
        per_point_sq_penalty = Norm(dif, self.inv_metric) 
        per_point_penalty = apply_fn(tf.sqrt, per_point_sq_penalty)

        # and average over all points
        return apply_fn(tf.reduce_mean, per_point_penalty, axis=0)


    # we allow an arbitrary boundary weight function which returns some weight depending on which
    # epoch we are on. this allows us to focus on the boundary at different points through training
    def train(
        self, 
        U : Model, 
        domain : tf.Tensor, # domain over which to train. should be a collection of B points [B, N]
        boundary : tf.Tensor, # boundary over which to train, likewise a collection of b points [b, N]
        optimizer : Optimizer, 
        epochs = 10,
        boundary_weight : Function | tf.Tensor = half # boundary weight function, should either be a scalar or return a scalar
    ):
        # we put a nice little progress bar for prettiness
        bar = Progbar(epochs, stateful_metrics=['action', 'boundary penalty'])

        action = self.action(U)
        boundary_penalty = self.boundary_penalty(U)
        
        # if boundary_weight is a constant - just go about as normal
        if not callable(boundary_weight): 
            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    A = action(domain)
                    B = boundary_penalty(boundary)

                    loss_value = convex_combo(tf.abs(A), B, boundary_weight)

                # apply gradients
                gradients = tape.gradient(loss_value, U.trainable_variables)
                optimizer.apply_gradients(zip(gradients, U.trainable_variables))  

                bar.update(epoch + 1, values=[('action', A), ('boundary penalty', B), ('loss', loss_value.numpy())]) 

        # otherwise we evaluate boundary_weight per-epoch.
        else:
            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    A = action(domain)
                    B = boundary_penalty(boundary)
                    loss_value = convex_combo(A, B, boundary_weight(epoch))

                # apply gradients
                gradients = tape.gradient(loss_value, U.trainable_variables)
                optimizer.apply_gradients(zip(gradients, U.trainable_variables))  

                bar.update(epoch + 1, values=[('action', A), ('boundary penalty', B), ('loss', loss_value.numpy())]) 


