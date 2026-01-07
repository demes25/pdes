# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Systems
# TODO


from context import *
from keras.optimizers import Optimizer
from keras.utils import Progbar
from keras.models import Model
from networks import NetworkOperator
from lattices import Domain, Image, Integral
from operators import Operator
from distributions import Distribution
from typing import Callable

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
        # TODO: add time-dependencies
        spatial_dims, # how many dimensions do we have?

        # the operator that takes our desired solution to zero
        operator : Operator,

        # a distribution on domains that gives us the image of the forcing term
        # if none, then the forcing term is assumed to be zero.
        forcing_term : Distribution | None = None,

        # the function we apply pointwise to the results of applying the desired operator, before integrating 
        # over the domain.
        #
        # by default, we square the value at each point
        # this is mainly to assert consistency in sign, so decreasing the integrated operator loss is
        # equivalent to decreasing the total absolute pointwise deviation from the correct solution.
        #
        # if the pointwise loss is None, we assume it is just the identity.
        pointwise_loss : Callable[[tf.Tensor], tf.Tensor] | None = tf.square,

        # a function that calculates the boundary penalty for some output image
        # if this is None, we apply no boundary penalty.
        boundary_penalty : Callable[[Image], tf.Tensor] | None = None, 
    ):
        self.spatial_dims = spatial_dims
        self.operator = operator

        self.forcing_term = forcing_term

        self.pointwise_loss = pointwise_loss
        self.boundary_penalty = boundary_penalty

    
    # sets a new forcing term
    def force(self, forcing_term : Distribution):
        self.forcing_term = forcing_term
    

    # we calculate the operator loss for some 
    # proposed solution image U
    #
    # we want Operator[U] = F, with F the forcing term
    # 
    # so we will define the loss image to be (Operator[U] - F)
    # and take the integral over the domain.
    #
    # if F is None, then we assume the equation Operator[U] = 0
    # and act accordingly.
    def operator_loss(self, solution_image : Image, forcing_image : Image | None = None) -> tf.Tensor:
        # we apply the operator:
        loss_image = self.operator(solution_image)

        # loss_image is the image holding the pointwise loss of the difference between Operator[U] and F
        # if F is None, then the loss image is just pointwise loss of Operator[U]

        # first we find the difference
        loss_grid = loss_image.grid if forcing_image is None else (loss_image.grid - forcing_image.grid)
        
        # then we apply the pointwise loss
        if self.pointwise_loss is not None:
            loss_grid = self.pointwise_loss(loss_grid)

        loss_image.grid = loss_grid

        return Integral(loss_image, average=True) # we integrate the loss over the domain and return.
    


    # we allow an arbitrary boundary weight function which returns some weight depending on which
    # epoch we are on. this allows us to focus on the boundary at different points through training
    def train(
        self, 
        U : Model, # U is the model we want to train. It must be wrapped as an operator
        domain : Domain, # domain over which to train. should be a collection of B points [B, N]
        optimizer : Optimizer, 
        epochs = 10,
        boundary_weight : Callable[[int], float | tf.Tensor] | tf.Tensor = half, # boundary weight function, should either be a scalar or return a scalar
        dynamic : bool = False, # allow the forcing_image to be created at each iteration from scratch
        noise : Callable[[tf.Tensor, tf.Tensor], Operator] | None = None
    ):
        # we put a nice little progress bar for prettiness
        bar = Progbar(epochs, stateful_metrics=['operator loss', 'boundary penalty'])

        # because i wrote this before I figured out that the memory couldn't take it,
        # i must now jump through loops to make dynamic forcing-image calculation possible. yay.
        if dynamic:
            # if dynamic, calculates from scratch each call
            def forcing_image():
                f = self.forcing_term(domain)
                return f if noise is None else noise(f)
        else:
            # otherwise, stores a copy in-scope.
            f = self.forcing_term(domain)
            f = f if noise is None else noise(f)
            def forcing_image():
                return f
            
        modelOperator = NetworkOperator(U)

        if self.boundary_penalty is None:
            # boundary_weight is ignored
            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    # we apply our model
                    f_ = forcing_image()
                    solution_image = modelOperator(f_)
                    loss_value = self.operator_loss(solution_image=solution_image, forcing_image=forcing_image()) 

                del solution_image

                # apply gradients
                gradients = tape.gradient(loss_value, U.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm = one) # clip gradients to norm 1
                optimizer.apply_gradients(zip(gradients, U.trainable_variables))  
                
                del gradients 
                del tape
                
                bar.update(epoch + 1, values=[('operator loss', loss_value.numpy()), ('loss', loss_value.numpy())])

        # if boundary_weight is a constant - just go about as normal
        elif not callable(boundary_weight): 
            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    # we apply our model
                    solution_image = modelOperator(forcing_image)
                    A = self.operator_loss(solution_image=solution_image, forcing_image=forcing_image) 
                    B = self.boundary_penalty(solution_image)
                    loss_value = convex_combo(A, B, boundary_weight)

                # apply gradients
                gradients = tape.gradient(loss_value, U.trainable_variables)
                optimizer.apply_gradients(zip(gradients, U.trainable_variables))  

                bar.update(epoch + 1, values=[('operator loss', A), ('boundary penalty', B), ('loss', loss_value.numpy())]) 

        # otherwise we evaluate boundary_weight per-epoch.
        else:
            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    solution_image = modelOperator(forcing_image)
                    A = self.operator_loss(solution_image=solution_image, forcing_image=forcing_image) 
                    B = self.boundary_penalty(solution_image)
                    loss_value = convex_combo(A, B, boundary_weight(epoch))

                # apply gradients
                gradients = tape.gradient(loss_value, U.trainable_variables)
                optimizer.apply_gradients(zip(gradients, U.trainable_variables))  

                bar.update(epoch + 1, values=[('operator loss', A), ('boundary penalty', B), ('loss', loss_value.numpy())]) 


