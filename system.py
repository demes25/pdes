# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Systems
# TODO


from context import *
from keras.optimizers import Optimizer
from keras.utils import Progbar
from keras.models import Model
from networks import OperatorWrapper

from operators import Operator, Domain, Image
from distributions import Distribution, Integral
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

        # an operator on images that gives us the forcing term
        forcing_term : Distribution,

        # the function we apply pointwise to the results of applying the desired operator.
        # by default, we square the value at each point and reduce to the mean
        pointwise_loss : Callable[[tf.Tensor], tf.Tensor] = tf.square,
        # a function that calculate the boundary penalty for some output image
        boundary_penalty : Callable[[Image], tf.Tensor] | None = None, 
    ):
        self.spatial_dims = spatial_dims
        self.operator = operator

        self.forcing_term = forcing_term

        self.pointwise_loss = pointwise_loss
        self.boundary_penalty = boundary_penalty

    
    # we calculate the operator loss for some 
    # proposed solution image U
    #
    # we want Operator[U] = F
    # 
    # so we will define the loss image to be (Operator[U] - F)
    # and take the integral over the domain.
    def operator_loss(self, solution_image : Image, forcing_image : Image) -> tf.Tensor:
        # we apply the operator:
        operator_image = self.operator(solution_image)

        # loss_image is the image holding the difference between Operator[U] and F
        loss_image = operator_image._mutate(new_mesh=self.pointwise_loss(operator_image.mesh-forcing_image.mesh))

        return Integral(loss_image) # we square each point-deviation from zero and return
    


    # we allow an arbitrary boundary weight function which returns some weight depending on which
    # epoch we are on. this allows us to focus on the boundary at different points through training
    def train(
        self, 
        U : Model, # U is the model we want to train. It must be wrapped as an operator
        domain : Domain, # domain over which to train. should be a collection of B points [B, N]
        optimizer : Optimizer, 
        epochs = 10,
        boundary_weight : Callable[[int], float | tf.Tensor] | tf.Tensor = half # boundary weight function, should either be a scalar or return a scalar
    ):
        # we put a nice little progress bar for prettiness
        bar = Progbar(epochs, stateful_metrics=['operator loss', 'boundary penalty'])
       
        base_image = Image(domain, pad=32)
        forcing_image = self.forcing_term(base_image)
       
        modelOperator = OperatorWrapper(U)

        if self.boundary_penalty is None:
            # boundary_weight is ignored
            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    # we apply our model
                    solution_image = modelOperator(forcing_image)
                    loss_value = self.operator_loss(solution_image=solution_image, forcing_image=forcing_image) 

                # apply gradients
                gradients = tape.gradient(loss_value, U.trainable_variables)
                optimizer.apply_gradients(zip(gradients, U.trainable_variables))  

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


