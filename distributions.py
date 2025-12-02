# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Distributions

import tensorflow as tf
from context import *
from typing import Callable
from lattices import Domain, Image, Integral
from functions import *


# --- DISTRIBUTIONS ---- #

# I will define this typehint, though it is in reality the same exact thing as Operator typehint.
# however we will be using it in a different context.
# 
# distributions act a function (or distribution generator) on the mesh
# while operators convolve differential kernels.
Distribution = Callable[[Domain], Image]


# distribution which simply acts some Function (i.e. [...] -> tf.Tensor) on the mesh of a given image.
def FunctionDistribution(fxn : Function, normalize : bool = True, scale : tf.Tensor = one) -> Distribution:
    def _h(x : Domain) -> Image:
        X = x.view(padded=True, flattened=True)
        f = fxn(X)

        func_shape = tf.shape(f)[1:].numpy().tolist()
        
        # if the function is a scalar:
        # then we adjust func_shape to be []
        if len(func_shape) == 0 or (len(func_shape) == 1 and func_shape[0] == 1):
            total_shape = x.leading_shape
        else:
            total_shape = tf.concat([x.leading_shape, func_shape], axis=0)

        func_grid = tf.reshape(f, shape=total_shape)

        func_image = Image(domain=x, grid=func_grid, shape=total_shape, entry_shape=func_shape)

        sc = scale 

        # if we normalize the image,
        # we divide it by its integral over the domain
        if normalize:
            integral = Integral(func_image)
            sc = sc / integral
        
        # note this function is scaled at will by some other scalar if we want it to be.
        # by default this will just be one.
        func_image.grid = func_image.grid * sc
        
        return func_image 
    
    return _h 
    

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
def Gaussian(mean : tf.Tensor, variance : tf.Tensor, normalize = True, scale : tf.Tensor = one) -> Distribution:
    # variance should be a scalar, 
    # mean should have dimension [N] same as each coordinate entry.
    
    # TODO: tend to the variance==0 case.
    if variance == zero:
        def _h(x : Image) -> Image:
            # if the variance is zero,
            # then we apply a 'dirac delta'

            # we first expand the means 
            mu = tf.expand_dims(mean, axis=0)

            # we identify the point on the mesh that is closest to this point,
            # and we make it so that its integral is one.
            g = x.geometry[0]
            flat_g = tf.reshape(g, shape=[-1, x.domain.dimension, x.domain.dimension])

            # we take x to be 'contravariant' coordinate 'vectors'
            x_flat = x.view(padded=True, flattened=True)
            
            dists = norm_sq_fn(flat_g)(x_flat - mu) # [B]
            index = tf.argmin(dists)
            
            value = tf.linalg.det(flat_g[index]) * tf.reduce_prod(x.domain.steps, axis=0) # to make this integrate to one, we divide by sqrt_g dXdY...
            new_flat = tf.one_hot(indices=index, depth=tf.size(dists), on_value=scale/value)
            
            new_mesh = tf.reshape(new_flat, shape=x.padded_mesh_shape)

            return Image(domain=x.domain, mesh=new_mesh, shape=x.padded_mesh_shape, geometry=x.geometry, pad=x.pad)

        return _h
    
    else:
        return FunctionDistribution(gaussian_fn(mean, variance), normalize=normalize, scale=scale)
        

# we apply a radial reciprocal from the given point
# i.e. 1/sqrt(dx^2 + dy^2 + ...)
#
# if the argument is zero here, we put a very large number.
#
# x should be [B, N]
# center should be [N]
# epsilon should be [] -- the small term we use to offset division by zero.
def Reciprocal(center : tf.Tensor, epsilon : tf.Tensor = 1e-4, normalize=True, scale : tf.Tensor = one) -> Distribution:
    return FunctionDistribution(reciprocal_fn(center=center, epsilon=epsilon), normalize=normalize, scale=scale)



def Sine(wavenum : tf.Tensor, normalize=True, scale : tf.Tensor = one) -> Distribution:
    return FunctionDistribution(sine_fn(wavenum), normalize=normalize, scale=scale)


