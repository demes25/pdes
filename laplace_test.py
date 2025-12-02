# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Laplacian Test

from operators import *
from keras.models import save_model, load_model
from keras import optimizers
from networks import SpatialKernel, NetworkOperator
from geometry import Euclidean
from lattices import Domain
from distributions import Distribution, Gaussian, Reciprocal, Sine
from system import System
from plots import save_heatmap

MODELPATH = None
NAME = 'test5'

# we test our current abilities.
# i'll try to find the derivative and second derivative of the square function
# using our mesh+convolution framework.
step = 0.01
padding = 32
# now we find the relevant ranges by looking at our desired side length 128
# (128 - 2 * 16) * 0.05 = 4.8
# so do:
start = -6.4
end = 6.4

# we want each point to look like [B, N]
dX = tf.constant([step, step])
ranges = tf.constant([[start, start], [end, end]])
domain = Domain(Euclidean(2), ranges, dX, padding=padding, dynamic=True) # this is now [b, b, 1])

k = pi/12.8 # our size is 12.8

# we now try training over many different forcing terms
forcing_terms : List[Distribution] = [
    Reciprocal([-0.5, 0.0], scale=one),
    Gaussian([-1.2, 0.4], 0.0125, scale=one),
    Sine(k, normalize = False, scale= -k*k),

    Reciprocal([0.2, -1.4], scale=two),
    Gaussian([-0.3, 1.6], 0.0175, scale=two),
    Sine(k*2, normalize = False, scale= -4*k*k),

    Reciprocal([-0.8, 1.0], scale=half),
    Gaussian([0.9, 0.7], 0.0225, scale=half),
    Sine(k*3, normalize = False, scale= -9*k*k),

    Reciprocal([-0.5, -1.9], scale=one),
    Gaussian([0.1, 0.5], 0.0275, scale=two),
    Sine(k*4, normalize = False, scale= -16*k*k)
]


if MODELPATH is None:
    system = System(2, operator=FlatSpatialLaplacian, pointwise_loss=tf.square)

    model = SpatialKernel(shape=[], dims=2, size=11, depth=3, init_filters=32, activation='relu')

    for f in forcing_terms:
        system.force(f)
        system.train(model, domain, optimizers.AdamW(), epochs=18, dynamic=True)
else:
    model = load_model(MODELPATH)

solution_operator = NetworkOperator(model)

i = 0
for forcing_term in forcing_terms:
    # we save the forcing image
    forcing_image = forcing_term(domain)
    save_heatmap(forcing_image.get(), f'logs/laplace/{NAME}/imgs/{i}-forcing_term.png')

    # the learned solution
    solution_image = solution_operator(forcing_image)
    save_heatmap(solution_image.get(), f'logs/laplace/{NAME}/imgs/{i}-solution.png')

    # and the "reported" forcing term - i.e. what we get
    # when we apply the desired operator on the learned solution
    laplacian_image = FlatSpatialLaplacian(solution_image)
    save_heatmap(laplacian_image.get(), f'logs/laplace/{NAME}/imgs/{i}-reconstructed_forcing_term.png')

    i+=1

if MODELPATH is None:
    save_model(model, f'logs/laplace/{NAME}/model.keras')