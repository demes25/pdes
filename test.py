# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Test

from operators import *
from keras.models import save_model, load_model
from keras.optimizers import AdamW
from geometry import Euclidean


# we test our current abilities.
# i'll try to find the derivative and second derivative of the square function
# using our mesh+convolution framework.
step = 0.05
start = 1.0
end = 20.0 


# we'll take the value to be 0 at t=0 and x=0
# we want each point to look like [B, 1]

dX = tf.constant([step])
ranges = tf.constant([[start], [end]])
domain = Domain(Euclidean(1), ranges, dX) # this is now [b, b, 1]

base_image = Image(domain, pad=2, all_around=False)
base_mesh = tf.squeeze(base_image.view())

square_image = base_image.apply(tf.square, mutable=True)
square_mesh = tf.squeeze(square_image.view())

grad = Gradient(square_image)
grad_mesh = tf.squeeze(grad.view())

laplace = ScalarLaplacian(square_image)
laplace_mesh = tf.squeeze(laplace.view())

result = tf.stack([square_mesh, grad_mesh, laplace_mesh], axis=1)

# i will make a plot here as well:

import tfplot

@tfplot.autowrap(figsize=(20, 11))
def plot(x, y, fig=None, ax=None):
    ax.plot(x, y)
    return fig

pl = plot(base_mesh, result)
tf.io.write_file("plot.png", tf.io.encode_png(pl))
