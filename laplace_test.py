# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Helmholtz Test

from operators import *
from keras.models import save_model, load_model
from keras.optimizers import AdamW
from networks import SpatialKernel, OperatorWrapper
from geometry import Euclidean
from distributions import Gaussian
from system import System


# we test our current abilities.
# i'll try to find the derivative and second derivative of the square function
# using our mesh+convolution framework.
step = 0.05
start = -6.4 # the size of the mesh should be divisible by 2 at least thrice
end = 6.4


# we'll take the value to be 0 at t=0 and x=0
# we want each point to look like [B, 1]
dX = tf.constant([step, step])
ranges = tf.constant([[start, start], [end, end]])
domain = Domain(Euclidean(2), ranges, dX) # this is now [b, b, 1])

base_image = Image(domain, pad=32)

forcing_term = Gaussian([1.0, 1.0], 0.0, scale=two)
forcing_image = forcing_term(base_image)

system = System(2, operator=ScalarLaplacian, forcing_term=forcing_term, pointwise_loss=tf.abs)

model = SpatialKernel(shape=[], dims=2, size=15, activation='relu')

system.train(model, domain, AdamW(), epochs=100)

save_model(model, 'laplace2d.keras')

solution_operator = OperatorWrapper(model)

solution_image = solution_operator(forcing_image)
solution_mesh = solution_image.view() # [X, Y]

'''
# do helmholtz operator on this 
operator_image = operator(solution_image)
operator_mesh = operator_image.view()

# pick some slice along Y
solution_slice = solution_mesh[:, 12]
axis_slice = base_mesh[:, 12, 0]
operator_slice = operator_mesh[:, 12]

result_slice = tf.stack([solution_slice, operator_slice], axis=-1)
'''


# i will make a plot here as well:
import tfplot

@tfplot.autowrap(figsize=(11, 11))
def heatmap(vals, fig=None, ax=None):

    im = ax.imshow(vals, extent=[-10.0, 10.0, -10.0, 10.0], cmap="viridis", origin="lower")

    # Optional colorbar
    fig.colorbar(im, ax=ax)
    return fig

pl = heatmap(solution_mesh)
tf.io.write_file("laplace_map_rh--dx=0_05--k=15.png", tf.io.encode_png(pl))
