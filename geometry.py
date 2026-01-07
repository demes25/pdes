# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Spacetimes

from context import *
from typing import Tuple
from functions import Function

# define some geometries

# each batched tensor will be assumed to look like [B, N, ...]
# where N is the possible amount of components in a given geometry.

# metrics should be tensor-to-tensor functions,
# which take in a batched rank 1 tensors (coordinates) and return batched rank-2 tensors

# geometries should be functionals which take in some parameter (if needed)
# and return a tuple -- (metric, inverse metric) -- of metric functions
# (note that this does not coincide exactly with what a geometry actually is - our functionals will also assume a fixed coordinate map)

# this is a type which represents our geometry
# -- i.e. a pair of functions denoting (g, g^-1) 
Geometry = Tuple[Function, Function] 


# some generic tensors:

# identity tensor
def Delta(N) -> tf.Tensor:
    return tf.linalg.diag(tf.ones([N], dtype=DTYPE))

# minkowski tensor - signature (-, +, +, +, ...)
def Eta(N) -> tf.Tensor:
    return tf.linalg.diag(tf.constant([-1] + [1]*(N-1), dtype=DTYPE))



# --- SPACES --- #

# takes in dimensionality N
# returns 2-tuple of constant functions fn : [B, N] -> B ** delta
#
# defined for any N
def Euclidean(
    N # dimensionality
) -> Geometry:
    
    _delta = Delta(N)

    # we expand dimensions for tiling purposes
    _delta = tf.expand_dims(_delta, axis=0)

    # X must be [B, N]
    # gives metric/inverse value(s) at X
    def _metric(X : tf.Tensor) -> tf.Tensor:
        B = tf.shape(X)[0]
        return tf.tile(_delta, (B, 1, 1))

    return (_metric, _metric)

# takes in scalar
# returns a tuple of fn : [B, N] -> [B, N, N]
#
# defined for N = 2 (polar) and N = 3 (standard spherical)
def Spherical(
    N # dimensionality
) -> Geometry: 
    
    if N == 2:
        # we define polar coordinates

        # X must be [B, N]
        # gives metric value(s) at X
        def _metric(X : tf.Tensor) -> tf.Tensor:
            # [r, theta]
            R = X[:, 0]

            ones = tf.ones_like(R, dtype=DTYPE)
            rsq = R * R
            diag_elements = tf.stack([ones, rsq], axis=-1)
            return tf.linalg.diag(diag_elements)
        
        # X must be [B, N]
        # gives inverse metric value(s) at X
        def _inverse_metric(X : tf.Tensor) -> tf.Tensor:
            # [r, theta]
            R = X[:, 0]
            
            ones = tf.ones_like(R, dtype=DTYPE)
            rsq = R * R
            diag_elements = tf.stack([ones, ones/rsq], axis=-1)
            return tf.linalg.diag(diag_elements)
    
    elif N == 3:
        # we construct spherical coordinates

        # X must be [B, N]
        # gives metric value(s) at X
        def _metric(X : tf.Tensor) -> tf.Tensor:
            # [r, theta, phi]
            R = X[:, 0]
            Theta = X[:, 1]

            # TODO: maybe extend to arbitrary dimensions in spherical spatial coordinates?
            # right now just for N=4
            ones = tf.ones_like(R, dtype=DTYPE)
            rsq = R * R
            sinth = tf.sin(Theta)
            diag_elements = tf.stack([ones, rsq, rsq * sinth * sinth], axis=-1)

            return tf.linalg.diag(diag_elements)

        # X must be [B, N]
        # gives inverse metric value(s) at X
        def _inverse_metric(X : tf.Tensor) -> tf.Tensor:
            # [r, theta, phi]
            R = X[:, 0]
            Theta = X[:, 1]

            # TODO: maybe extend to arbitrary dimensions in spherical spatial coordinates?
            # right now just for N=4
            ones = tf.ones_like(R, dtype=DTYPE)
            rsq = R * R
            sinth = tf.sin(Theta)
            diag_elements = tf.stack([ones, ones/rsq, ones/(rsq * sinth * sinth)], axis=-1)

            return tf.linalg.diag(diag_elements)

    else:
        #TODO: generalize spherical-like metrics
        raise NotImplemented("Spherical metric only implemented for dimensions 2 and 3")

    return (_metric, _inverse_metric)
    



# --- SPACETIMES --- #

# takes in None
# returns 2-tuple of constant functions fn : [B, N] -> B ** eta
#
# defined for any N
def Minkowski(
    N = 4 # dimensionality
) -> Geometry:
      
    _eta = Eta(N)

    # we expand dimensions for tiling purposes
    _eta = tf.expand_dims(_eta, axis=0)

    # X must be [B, N]
    # gives metric/inverse value(s) at X
    def _metric(X : tf.Tensor) -> tf.Tensor:
        B = tf.shape(X)[0]
        return tf.tile(_eta, (B, 1, 1))
    
    return (_metric, _metric)

# takes in scalar
# returns a tuple of fn : [B, N] -> [B, N, N]
#
# defined for N = 4 only
def Schwarzschild(
    M, # black hole mass
    N = 4 # dimensionality
) -> Geometry:
    
    #TODO: generalize Schwarzschild-like metrics
    if N != 4:
        raise NotImplemented("Schwarzschild metric only implemented for dimension 4")

    schw_radius = tf.constant(2 * M, dtype=DTYPE) * G/(c*c)

    # X must be [B, N]
    # gives metric value(s) at X
    def _metric(X : tf.Tensor) -> tf.Tensor:
        # [t, r, theta, phi]
        R = X[:, 1]
        Theta = X[:, 2]

        # TODO: maybe extend to arbitrary dimensions in spherical spatial coordinates?
        # right now just for N=4
        ones = tf.ones_like(R, dtype=DTYPE)

        U = ones - schw_radius/R #  
        rsq = R * R
        sinth = tf.sin(Theta)

        diag_elements = tf.stack([-U, ones/U, rsq, rsq * sinth * sinth], axis=-1)

        return tf.linalg.diag(diag_elements)

    
    # X must be [B, N]
    # gives inverse metric value(s) at X
    def _inverse_metric(X : tf.Tensor) -> tf.Tensor:
        # [t, r, theta, phi]
        R = X[:, 1]
        Theta = X[:, 2]

        # TODO: maybe extend to arbitrary dimensions in spherical spatial coordinates?
        # right now just for N=4
        ones = tf.ones_like(R, dtype=DTYPE)

        U = ones - schw_radius/R #  
        rsq = R * R
        sinth = tf.sin(Theta)

        diag_elements = tf.stack([-ones/U, U, ones/rsq, ones/(rsq * sinth * sinth)], axis=-1)

        return tf.linalg.diag(diag_elements)
        
    return (_metric, _inverse_metric)



