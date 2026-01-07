# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Operator Test

from operators import *
from distributions import Distribute
from geometry import Euclidean
from lattices import Domain 
from plots import save_plot

# we test our current abilities.
# i'll try to find the derivative and second derivative of the square function
# using our mesh+convolution framework.
step = 0.05
start = -10.0
end = 10.0 


# we'll take the value to be 0 at t=0 and x=0
# we want each point to look like [B, 1]
dX = tf.constant([step])
ranges = tf.constant([[start], [end]])
domain = Domain(Euclidean(1), ranges, dX, padding=2) # this is now [b, b, 1])

base_mesh = tf.squeeze(domain.view())

square_image = Distribute(tf.square, normalize=False)(domain)
square_mesh = tf.squeeze(square_image.view())

grad = Gradient(square_image)
grad_mesh = tf.squeeze(grad.view())

laplace = FlatSpatialLaplacian(square_image)
laplace_mesh = tf.squeeze(laplace.view())

result = tf.stack([square_mesh, grad_mesh, laplace_mesh], axis=1)

save_plot(base_mesh, result, 'square_plot_2.png')