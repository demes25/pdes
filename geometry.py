# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Geometry

from operators import *

# Here we will define some geometric notions such as metrics and covariant derivatives

# set the amount of components to 4 (spacetime)
N = 4

# physical constants
# I have set these to natural units
G = one 
c = one 
h = one

# define some metrics/geometries

# each batched tensor will be assumed to look like [B, N, ...]

# metrics should be tensor-to-tensor functions,
# which takes in a batched rank 1 tensor (coordinates) and returns a batched rank-2 tensor

# geometries should be functionals which take in some parameter (if needed)
# and return a tuple -- (metric, inverse metric) -- of metric functions
# (note that this does not coincide exactly with what geometry actually is - our functionals will also assume a fixed coordinate map)

# identity tensor
_delta = tf.linalg.diag(tf.ones([N], dtype=DTYPE))

# minkowski metric - signature (-, +, +, +)
_eta = tf.linalg.diag(tf.constant([-1] + [1]*(N-1), dtype=DTYPE))

# takes in None
# returns 2-tuple of constant functions fn : [B, N] -> B ** eta
def MinkowskiGeometry():
    # X must be [B, N]
    def _m(X):
        B = tf.shape(X)[0]
        return tf.tile(_eta, (B, N, N))
    
    return (_m, _m)

# takes in []
# returns a tuple of fn : [B, N] -> [B, N, N]
# (g, g^-1)
def SchwarzschildGeometry(mass):
    schw_radius = tf.constant(2 * mass, dtype=DTYPE) * G/(c*c)

    # X must be [B, N]
    def _metric(X):
        # [t, r, theta, phi]
        R = X[:, 1]
        Theta = X[:, 2]

        # TODO: maybe extend to arbitrary dimensions in spherical spatial coordinates?
        # right now just for N=4
        U = one - schw_radius/R #  
        rsq = R * R
        sinth = tf.sin(Theta)

        diag_elements = tf.stack([-U, one/U, rsq, rsq * sinth * sinth], axis=-1)

        return tf.linalg.diag(diag_elements)

    
    # X must be [B, N]
    def _inv_metric(X):
        # [t, r, theta, phi]
        R = X[:, 1]
        Theta = X[:, 2]

        # TODO: maybe extend to arbitrary dimensions in spherical spatial coordinates?
        # right now just for N=4
        U = one - schw_radius/R #  
        rsq = R * R
        sinth = tf.sin(Theta)

        diag_elements = tf.stack([-one/U, U, one/rsq, one/(rsq * sinth * sinth)], axis=-1)

        return tf.linalg.diag(diag_elements)
        
    return (_metric, _inv_metric)



# some geometric notions as well

# U, V both [B, N]
# returns inner product [B]
def InnerProducts(U, V, g=None):
    # if g is none, we assume that we want straight-up dot products
    # so do exactly that
    if g is None:
        prod = mul_fn(U, V)
        return apply_fn(tf.reduce_sum, prod, axis=-1)
    
    # otherwise, we want einstein summation
    def _bilinear(u, a, v):
        # we write einstein summation, where b is the batch index and i/j are dummy
        return tf.einsum('bi,bij,bj->b', u, a, v)
    
    return apply_fn(_bilinear, U, g, V)

# returns InnerProducts(U, U)
def Norms(U, g=None):
    return InnerProducts(U, U, g=g)