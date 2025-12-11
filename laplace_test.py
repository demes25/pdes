# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Laplacian Test

from operators import *
from keras.models import save_model, load_model
from keras import optimizers
from networks import SpatialKernel, NetworkOperator
from metrics import RR
from geometry import Euclidean
from lattices import Domain
from distributions import Distribution, Gaussian, Reciprocal, Sine, BatchDistribute
from functions import *
from system import System
from plots import save_heatmap


MODELPATH = None #'logs/laplace/test5/model.keras'
SAVE_IMGS = True
NAME = 'test6'

# we test our current abilities.
# i'll try to find the derivative and second derivative of the square function
# using our mesh+convolution framework.
step = 0.05
epsilon = step*step 

padding = 32
# now we find the relevant ranges by looking at our desired side length 256
# (256 - 2 * 16) * 0.05 = 4.8
# so do:
start = -4.8
end = 4.8

# we want each point to look like [B, N]
dX = tf.constant([step, step])
ranges = tf.constant([[start, start], [end, end]])
domain = Domain(Euclidean(2), ranges, dX, padding=padding, dynamic=True) # this is now [b, b, 1])

k = pi/9.6 # our size is 9.6

# we now try training over many different forcing terms
#def forcing_term(center, var, scale, wavenum):
#    return BatchDistribute([reciprocal_fn(center, epsilon=epsilon), gaussian_fn(center, var), sine_fn(wavenum)], normalize=False, scale=scale)

params = [([-0.5, 0.0], 0.02), ([1.0, -0.4], 0.03), ([2.1, 0.1], 0.04), ([2.3, 1.7], 0.05), ([1.3, 4.3], 0.06)]

forcing_terms : List[Distribution] = [
    *[Gaussian(x, y, scale=two)
    for x, y in params],
    #*[Reciprocal(x) for x, _ in params]
]


if MODELPATH is None:
    system = System(2, operator=FlatSpatialLaplacian, pointwise_loss=tf.square)

    model = SpatialKernel(shape=[], dims=2, size=5, depth=5, init_filters=32, activation='relu')

    for i in range(3):
        for f in forcing_terms:
            system.force(f)
            system.train(model, domain, optimizers.AdamW(learning_rate=1e-5), epochs=3, dynamic=True)
else:
    model = load_model(MODELPATH)

solution_operator = NetworkOperator(model)

i = 0

rsq = []

for forcing_term in forcing_terms:
    # we save the forcing image
    forcing_image = forcing_term(domain)
    
    # the learned solution
    solution_image = solution_operator(forcing_image)
    # and the "reported" forcing term - i.e. what we get
    # when we apply the desired operator on the learned solution
    laplacian_image = FlatSpatialLaplacian(solution_image)
    
    

    if SAVE_IMGS:
        for j in range(forcing_image.batches):
            i += 1
            save_heatmap(forcing_image.get(j), f'logs/laplace/{NAME}/imgs/{i}-forcing_term.png')
            save_heatmap(laplacian_image.get(j), f'logs/laplace/{NAME}/imgs/{i}-reconstructed_forcing_term.png')
            save_heatmap(solution_image.get(j), f'logs/laplace/{NAME}/imgs/{i}-solution.png')

    rsq.append(RR(forcing_image, laplacian_image))

rsq = tf.reduce_mean(rsq, axis=0)

print(f"average R square value: {rsq.numpy()}")

if MODELPATH is None:
    save_model(model, f'logs/laplace/{NAME}/model.keras')