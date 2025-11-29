# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Distributions

import tensorflow as tf
from context import *
from typing import ParamSpec, Callable
from geometry import Function
from operators import Image, padding_adjuster
from functions import gaussian_fn, norm_sq_fn


# calculates the 'total integral' of the given image
# 
# i.e. the left riemann sum
#
# int(unpadded_domain)[ img(x) sqrt_g(x) dx ]
#
# we return a tensor of shape image.func_shape
#
# TODO: domain.steps may not be constant in the domain per discretized point
# also, their product may be different. work on generalizing to non-cartesian coordinates.
def Integral(image : Image) -> tf.Tensor: 
    flattened = image.view(flattened=True) 

    g = image.geometry[0][padding_adjuster(image.pad, image.domain.dimension, 2)]  # we get the unpadded metric
    flat_sqrt_g = tf.reshape(tf.sqrt(tf.abs(tf.linalg.det(g))), shape=[-1]) # we find the sqrt of det g

    dVol = flat_sqrt_g * tf.reduce_prod(image.domain.steps) # we multiply by dXdYdZ...

    new_shape = tf.concat([tf.shape(dVol), [1 for _ in image.func_shape]], axis=0)
    
    dVol = tf.reshape(dVol, new_shape)

    integrable = flattened * dVol

    return tf.reduce_sum(integrable, axis=0) 



# --- DISTRIBUTIONS ---- #

# I will define this typehint, though it is in reality the same exact thing as Operator typehint.
# however we will be using it in a different context.
# 
# distributions act a function (or distribution generator) on the mesh
# while operators convolve differential kernels.
Distribution = Callable[[Image], Image]


# we apply the gaussian
# 
# 1/sqrt(2pi var) exp((x-mu)^2/2var)

# mu is the mean, x is the coordinates at the given point in the mesh,
# and var is the variance.
# 
# note that the limit of this expression as we approach var=0 is the dirac delta.
# I will construct a similar discretized dirac delta for this.
# 
# x should be [B, N]
# mean should be [N]
# variance should be [] 
def Gaussian(mean : tf.Tensor, variance : tf.Tensor, normalize = True, scale : tf.Tensor = one, mutable=False) -> Distribution:
    # variance should be a scalar, 
    # mean should have the same dimensions as the coordinates.
    
    mean = tf.expand_dims(mean, axis=0)
    scale = scale 

    if variance == zero:
        def _h(x : Image) -> Image:
            # if the variance is zero,
            # then we apply a 'dirac delta'

            # we identify the point on the mesh that is closest to this point,
            # and we make it so that its integral is one.
            g = x.geometry[0]
            flat_g = tf.reshape(g, shape=[-1, x.domain.dimension, x.domain.dimension])

            # we take x to be 'contravariant' coordinate 'vectors'
            x_flat = x.view(padded=True, flattened=True)
            
            dists = norm_sq_fn(flat_g)(x_flat - mean) # [B]
            index = tf.argmin(dists)
            
            value = tf.linalg.det(flat_g[index]) * tf.reduce_prod(x.domain.steps, axis=0) # to make this integrate to one, we divide by sqrt_g dXdY...
            new_flat = tf.one_hot(indices=index, depth=tf.size(dists), on_value=scale/value)
            
            new_mesh = tf.reshape(new_flat, shape=x.padded_mesh_shape)

            return Image(domain=x.domain, mesh=new_mesh, shape=x.padded_mesh_shape, geometry=x.geometry, pad=x.pad)

        return _h
    
    else:
        gaussian = gaussian_fn(mean, variance)

        def _h(x : Image) -> Image:
            gaussian_img = x.apply(gaussian, mutable=mutable)

            # if we normalize the gaussian,
            # we divide it by its integral
            if normalize:
                integral = Integral(gaussian_img)
                scale = scale / integral
            
            # note this gaussian is scaled at will by some other scalar if we want it to be.
            # by default this will just be one.
            return gaussian_img._mutate(gaussian_img.mesh * scale)
        
        return _h
        
        





