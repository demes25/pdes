# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Operators


from context import *
from typing import Callable, Tuple, List, Sequence
from lattices import Image

    
# --- KERNELS --- #

# we define kernels for differential operators
# these will act on an N-dimensional mesh
#
# our meshes will look like [n1, n2, ..., nN, N]
# so our function outputs will look like [n1, n2, ..., nN, *dimF]
#
# this gives back the kernel corresponding to the i'th partial derivative
# when applied to our function outputs.

# we define "kernel bases" that look like (stencil tensor, shape)
KernelBase = Tuple[tf.Tensor, List[int]]

# we follow standard procedure, centered kernel:
#           df = [f(x+dx) - f(x-dx)]/2
# and also right-handed:
#           df = f(x+dx) - f(x)
#
# The partial bases are one-dimensional !!
right_partial_base : KernelBase = (tf.constant([-1, 1], dtype=DTYPE), [2])
center_partial_base : KernelBase = (tf.constant([-0.5, 0, 0.5], dtype=DTYPE), [3])

# we define the 2d flat second partial kernel:
second_partial_base : KernelBase = (tf.constant([1, -2, 1], dtype=DTYPE), [3])


# gives us the shape of the kernel acting on the given axis, based on the base shape and the mesh dimensions
def kernel_shape(base_shape : Sequence[int], dims : int, axis=0) -> tf.Tensor:
    sizes = [1] * dims
    
    for j in base_shape:
        sizes[axis] = j
        axis += 1
    
    return sizes


def CreateKernel(base : KernelBase, dims : int, func_size : int, axis=0)-> tf.Tensor:
    # this gives a differential kernel along the given axis
    # without dividing by the relevant dx.
    # so this takes in F and returns dF
    stencil, base_shape = base 

    sizes = kernel_shape(base_shape=base_shape, dims=dims, axis=axis)

    stencil = tf.reshape(stencil, shape=sizes + [1, 1])
    # reshape to [k1, k2, ..., kN, func_size, func_size]
    return tf.broadcast_to(stencil, shape=sizes + [func_size, func_size])
    
    
def ReshapeKernel(kernel : tf.Tensor, base : KernelBase, dims : int, func_size : int, axis=0)-> tf.Tensor:
    # we reshape the given kernel to act along the given  axis.
    sizes = kernel_shape(base_shape=base[1], dims=dims, axis=axis)
    return tf.reshape(kernel, shape=sizes + [func_size, func_size])
    


# --- some help --- #

# we follow standard boilerplate for acting the above differntial operators
# on some mesh
def prep_for_kernel(image : Image):
    # we expect an input channel dim and an output channel dim for convolution.
    # in this case we reshape so that the input channel dim is always 1.
    # but the output channel dim might change depending on the output shape of our function.
    
    # mesh_shape is the shape of the mesh - i.e. the whole [n1, n2, ..., N] leading up to everything
    # and func_shape is the output shape of the function.

    # in all, substrate shape should look like   mesh_shape + func_shape

    if image.rank == 0:
        # this will just be our default treatment of scalars (rank 0) - expand them to act as rank 1.
        val_size = 1
        # expand out so our function out size becomes 1
        substrate = tf.expand_dims(image.grid, axis=-1)
    else:
        val_size = tf.reduce_prod(image.entry_shape)
        
        # we compress our function output shape - we will later reshape this back
        reduction_shape = tf.concat([image.leading_shape, [val_size]], axis=0)
        substrate = tf.reshape(image.grid, shape=reduction_shape)

    # we return the prepped substrate,
    # and the relevant dim_out for convolution also
    return (substrate, val_size)


# --- OPERATORS --- # 

# now we define operators.
# these act between images.
Operator = Callable[[Image], Image]


# since many kernel bases are reused over multiple axes,
# we wrap the iteration in a nice function
def ConvolveIter(
    base : KernelBase,
    axes : Sequence[int] = [], # the list of axes over to convolve

    # if we want to apply a final function onto the new mesh before we 
    # return the image.
    # it is assumed to be a function of the final mesh tensor and the initial image.
    output_gate : Callable[[tf.Tensor, Image], tf.Tensor] | None = None 
) -> Operator:

    def _h(phi : Image) -> Image:
        domain = phi.domain
        dims = domain.dimension 

        substrate, func_size = prep_for_kernel(phi)
        outputs = []

        kernel = CreateKernel(base=base, dims=dims, func_size=func_size)

        # if axes are not specified, we calculate over all variables by default.
        for i in (axes if axes else range(domain.dimension)):
            kernel = ReshapeKernel(kernel=kernel, base=base, dims=dims, func_size=func_size, axis=i)
            
            # we pad convolutions to the same size as before, and account for this by including
            # sufficient padding beforehand.
            post_kernel = tf.nn.convolution(substrate, kernel, padding='SAME')
            output = tf.reshape(post_kernel, shape=phi.shape)
            outputs.append(output)
        
        # stack along the final axis
        final_grid = tf.stack(outputs, axis=-1)

        # apply the output gate if present
        if output_gate is not None:
            final_grid = output_gate(final_grid, phi)

        return Image(domain=domain, grid=final_grid, entry_shape=domain.entry_shape)
    
    return _h 


# this returns an operator that calculates the partial derivatives along the
# specified coordinates and stacks on the last axis. if empty, will calculate the full gradient.
#
# this is an iterative convolution over the specified axes in wrt[].
# the output gate is simply dividing by dx.
def Partials(
    wrt = [], # the list of coordinate indices over which to compute partials
    centered=False
) -> Operator:
    # returns tensor of relevant partial derivatives of f over the given domain, 
    # same coordinate order as in wrt, stacked along last axis.

    new_dim = len(wrt)

    # we define the output gate - divide by dx
    def output_gate(output_mesh : tf.Tensor, phi : Image) -> tf.Tensor:
        # this is [N]
        dX = phi.domain.steps 

        # or [M], call it.
        if wrt:
            dX = tf.gather(dX, indices=wrt, axis=0)
        
        dims = phi.dimension
        M = new_dim or dims
        
        # and the output mesh looks like [*mesh_shape, *func_shape, M]
        # so we reshape dX like so:
        dX = tf.reshape(dX, shape = [1] * (dims + phi.rank) + [M])

        return output_mesh/dX
    
    return ConvolveIter(
        base = center_partial_base if centered else right_partial_base,
        axes=wrt,
        output_gate=output_gate
    )


# the full gradient.
# takes in an image of rank n,
# returns one of rank n+1
Gradient : Operator = Partials()


# computes non-mixed second partials
def DiagonalSecondPartials(
    wrt = [], # the list of coordinate indices over which to compute second partials
) -> Operator:
    
    # returns tensor of relevant second partial derivatives of f over the given domain, 
    # same coordinate order as in wrt, stacked along last axis.

    new_dim = len(wrt)

    # we define the output gate - divide by dx^2
    def output_gate(output_mesh : tf.Tensor, phi : Image) -> tf.Tensor:
        # this is [N]
        dX = phi.domain.steps 

        # or [M], call it.
        if wrt:
            dX = tf.gather(dX, indices=wrt, axis=0)
        
        dims = phi.dimension
        M = new_dim or dims
        
        dX_sq = tf.square(dX)

        # and the output mesh looks like [*mesh_shape, *func_shape, M]
        # so we reshape dX like so:
        dX_sq = tf.reshape(dX_sq, shape = [1] * (dims + phi.rank) + [M])

        return output_mesh/dX_sq
    
    return ConvolveIter(
        base = second_partial_base,
        axes=wrt,
        output_gate=output_gate
    )

# the full main diagonal of the hessian matrix.
HessianDiagonal : Operator = DiagonalSecondPartials()


# the Laplacian operator in flat-space - sums the diagonal second partials
def FlatSpatialLaplacian(phi : Image) -> Image:
    partials_grid = HessianDiagonal(phi).grid
    # sum through the coordinate second partials to get the flat laplacian
    laplacian_grid = tf.reduce_sum(partials_grid, axis=-1)

    return Image(domain=phi.domain, grid=laplacian_grid, shape=phi.shape, entry_shape=phi.entry_shape)



# I am adding a noise operator that will introduce noise into the image
def NoiseOperator(mean : tf.Tensor = zero, stddev : tf.Tensor = one) -> Operator:
    def _h(x : Image) -> Image:
        noise = tf.random.normal(shape=x.shape, mean=mean, stddev=stddev, dtype=x.grid.dtype)

        return Image(domain=x.domain, grid=x.grid + noise, shape=x.shape, entry_shape=x.entry_shape, leading_shape=x.leading_shape, batches=x.batches)
    
    return _h


'''

# this is the vector divergence, div(A) = 1/sqrt(g) d_mu sqrt(g) A^mu
def VectorDivergence(
    covariant=True, # true if we must first raise indices
    
    mutable=False 
    # we need to apply gradients after we prepare the entries with raising (if needed) and by multiplying with the 'volume form'
    # if mutable is True, we will prepare the mesh of A in-place for gradients. otherwise, we will not alter the mesh of A and instead create a new temporary Image to take gradients.
) -> Operator:
    
    def _h(A : Image) -> Image:
        assert len(A.func_shape) == 1, f'{A} must be a vector field'
        # this is [B, N]
        flat_A = A.view(padded=True, flattened=True)
        
        # TODO: write a generalized raising/lowering function
        g, g_inv = A.geometry
        sqrt_g = tf.sqrt(tf.abs(tf.linalg.det(g))) # sqrt(|detg|)

        N = A.domain.dimension

        # we reshape to batch indices
        flat_sqrt_g = tf.reshape(sqrt_g, shape=[-1]) # [B]

        if covariant:
            # if A is covariant, we must raise.

            flat_g_inv = tf.reshape(g_inv, shape = [-1, N, N]) # [B, N, N]
            # to raise, we then einsum over the mesh
            flat_A = tf.einsum('bij,bj->bi', flat_g_inv, flat_A) # this is now [B, N]
        

        exp_sqrt_g = tf.expand_dims(flat_sqrt_g, axis=-1) # [B, 1]
        
        flat_A = flat_A * exp_sqrt_g # [B, N]
        # this is sqrt(g) partial^mu phi

        # now we must take gradients again, and then trace over mu.
        #
        # since we have not affected any metadata, (shape is same, padding is same)
        # we can just re-expand sqrt_g_raised and replace image.mesh by this new mesh,
        # and then apply the gradient operator

        new_mesh = tf.reshape(flat_A, A.shape)
        
        # if we are allowed to alter A in-place, then we replace the unprepped mesh with the prepped mesh 
        if mutable:
            A._mutate(new_mesh=new_mesh)
        else:
            # otherwise we make a new image
            A = Image(A.domain, new_mesh, A.shape, A.geometry, A.pad)

        # we take derivatives.
        derivs : Image = Gradient(A)

        # take again the flattened second derivatives, and contract
        flat_derivs = derivs.view(padded=True, flattened=True)
        
        flat_result = tf.einsum('bii->b', flat_derivs) # we have successfully contracted over the indices. this is now [B]
        
        # now we reshape to the mesh
        result_mesh = tf.reshape(flat_result, shape=derivs.padded_mesh_shape) 
        
        # the only thing left to do is divide by the 'volume form'
        # voila! we have the generalized laplacian/beltrami
        div_mesh = result_mesh/sqrt_g

        # now - the only difference between this and the hessian 
        # is the function shape. here it is [], but for the hessian it is [N, N]
        # 
        # we can adjust this in the same hessian object,
        # and return it.
        div = derivs._mutate(new_mesh=div_mesh, new_func_shape=[])

        return div

    return _h


# NOTE: this is NOISY for too small a dX. 
# consider other approaches, like a euclidean laplace stencil.
def ScalarLaplacian(phi : Image) -> Image:
    # we calculate the laplacian for a scalar field.
    # this looks like div(grad(phi))
    assert len(phi.func_shape) == 0, f'{phi} must be a scalar field'
    div : Operator = VectorDivergence(covariant=True, mutable=True)

    return div(Gradient(phi))



# We formulate some other useful operators.
def HelmholtzOperator(k : tf.Tensor): # k should be a scalar
    
    # phi should be a scalar also
    def _h(phi : Image) -> Image:
        assert len(phi.func_shape) == 0, f'{phi} must be a scalar field'

        # Laplace(phi) + k(phi) = 0
        laplace_phi = ScalarLaplacian(phi)

        k_phi_mesh = k * phi.mesh 

        return laplace_phi._mutate(new_mesh=laplace_phi.mesh + k_phi_mesh)
        
    return _h

'''