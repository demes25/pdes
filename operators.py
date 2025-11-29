# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Operators


from context import *
from typing import Callable, Tuple, List
from geometry import Function, Geometry


# --- DOMAINS --- #

# this is a wrapped tensor dataclass that can discretize a given range of values
# TODO: generalize to other discretization methods. right now it just makes a linear mesh
class Domain:
    def __init__(
        self,
        geometry : Geometry, # a pair of functions, each of which return [B, N, N]
        ranges : tf.Tensor, # should be a [2, N] tensor,
        steps : tf.Tensor, # should be a [N] tensor  

        pad = 0 # we allow padding around the boundary for convolution purposes
    ):
        
        # this is N
        self.dimension = int(tf.shape(ranges)[1].numpy())
        
        # TODO: make a 'convolvable mesh' which has padded dimensions for convolvability purposes.
        # or do some other workaround for the padding problem

        self.ranges = ranges 
        self.steps = steps

        # the unpadded shape !!
        self.shape = tf.cast((ranges[1] - ranges[0])/steps, tf.int32)

        self._dtype = tf.as_dtype(self.ranges.dtype)

        self.geometry = geometry

    # we discretize.
    #
    # the input is X [2, N] (the ranges) and dX [N] (the step sizes)
    #
    # where N is the dimension count, the first col tells us starting points and 
    # second col tells us ending points.
    #
    # dX tells us how far away points should be in each dimension.
    #
    # so this will give an [n**N, N] grid
    #
    # for now this splits our ranges linearly via tf.linspace
    # but this is slightly suspect for more nontrivial geometries, like spherical
    # still, though, should work not too bad.
    #
    # we use meshgrid to produce a cartesian product of linspaces
    # as such I will make the indexing adjustable depending on how we want it
    def discretize(self, pad=0, indexing='ij') -> tf.Tensor: 

        if pad == 0:
            X = self.ranges 
        else:
            padder = self.steps * pad
            # pad evenly all around
            padding = tf.stack([-padder, padder], axis=0)
            X = self.ranges + padding
        
        dX = self.steps

        grid_slices = []

        starts = X[0]
        ends = X[1]

        for i in range(tf.shape(starts)[0]):
            start = starts[i]
            end = ends[i]
            spacing = dX[i]
            
            # TODO: this can be made more general to take 'spacing' as literal distance
            # and use the metric at a given point to find the next discrete point by this measure of distance
            
            n = (end - start)/spacing # we find the amount of points
            n = tf.cast(n, tf.int32) # must be integer

            new_slice = tf.linspace(start, end, n) # from x[0] to x[1], n discrete points

            grid_slices.append(new_slice)
        
        grid = tf.meshgrid(*grid_slices, indexing=indexing)

        # then we stack along the last axis
        return tf.stack(grid, axis=-1)
    

# we create images - which are objects
# that wrap meshes and include their metadata (shape, padding, metric, etc)

# --- IMAGES --- #

# this is a helper function - it is a list of slices that throws away the 
# specified amount of padding.
def padding_adjuster(
        pad, # the amount of padding to "throw away" 

        mesh_len, # the amount of axes to cut
        # the leading axes are assumed to be the mesh axes

        func_len = 0 # the amount of axes not to cut (assumed scalar function, len = 0)
        # the ending axes are assumed to be the function axes

        # if we take the mesh to look like [n1, n2, ..., nN, f1, f2, ... fF]
        # where n1, n2 are the mesh axes and f1, f2... are the component axes of 
        # the function evaluated at the corresponding mesh points,
        # 
        # then mesh_len = N
        # and func_len = F
    ) -> List[slice]:

    if pad == 0:
        # if the padding is 0, we just return [:,:,:,...]
        return [slice(None) for _ in range(mesh_len+func_len)]
    
    # we return [pad:-pad, pad:-pad, ..., pad:-pad, :, :, ..., :]
    pad_slice = slice(pad, -pad)
    
    if func_len == 0:
        return [pad_slice for _ in range(mesh_len)]
    
    return [pad_slice for _ in range(mesh_len)] + [slice(None) for _ in range(func_len)]
    

class Image:
    def __init__(
        self,
        domain : Domain, # the domain

        mesh : tf.Tensor = None, # the mesh that we wrap here, a tensor of the shape [n1, n2, ..., nN, *F] where F is the shape of the mesh value at (x1, x2, ..., xN)
        
        shape : tf.Tensor = None, # by default, if shape and main are None, we discretize the domain points here and store the 'base mesh' 

        geometry : Tuple[tf.Tensor, tf.Tensor] | None = None, # we pass tensors of the metric and inverse metric evaluated at the domain mesh points.
        # this should be none  ^^ only if mesh is None

        pad = 0, # how much padding we have
    ):
        self.domain = domain
        
        if mesh is None:
            # if the mesh is None -- this becomes a coordinate image.
            # the discretized domain.
            self.mesh = domain.discretize(pad=pad)
        else:
            self.mesh = mesh

        if shape is None:
            shape = tf.shape(self.mesh)

        self.shape = shape 
        self.padded_mesh_shape = shape[:domain.dimension] # we find the shape of the padded mesh, we keep as a tensor. [n1 + 2p, n2 + 2p, n3 + 2p, ... nN + 2p]
        self.func_shape = shape[domain.dimension:].numpy().tolist() # we keep this as a list for manipulability, this is F from a bove

        # note that the unpadded mesh shape [n1, n2, n3, ..., nN] is stored in domain.shape 
        self.pad = pad

        # "unpads" the padded mesh
        self.unpadded_slicer = padding_adjuster(pad, mesh_len=domain.dimension, func_len = len(self.func_shape))

        if geometry is None:
            assert mesh is None, 'Parameter "geometry" may only be None if we are calculating the image directly of the domain points'
            # ^^ otherwise we must re-calculate the (hypothetically already calculated) mesh to find the metric tensors, which is costly 

            # functions expect something like [B, N]
            # so we use the flattened mesh
            x = tf.reshape(self.mesh, shape=[-1, domain.dimension])

            # we then apply the geometry functions
            g, g_inv = domain.geometry

            g = g(x) 
            g_inv = g_inv(x)
            # ^^ these are now both [B, N, N], evaluated at all points in the mesh

            # we now reshape each of them back:
            re_shape = tf.concat([self.padded_mesh_shape, [domain.dimension, domain.dimension]], axis=0)

            g = tf.reshape(g, shape=re_shape)
            g_inv = tf.reshape(g, shape=re_shape)

            # these are now [n1, n2, ..., nN, N, N]
            self.geometry = (g, g_inv)
        
        else:
            self.geometry = geometry 


    def view(self, padded=False, flattened=False) -> tf.Tensor:
        # we view the mesh.
        # this is good if we have padding - we can view with/without padding at will.
        mesh = self.mesh if padded else self.mesh[self.unpadded_slicer]

        # we can also flatten if needed, such as for function application
        return tf.reshape(mesh, shape=[-1, *self.func_shape]) if flattened else mesh

    def _mutate(self, new_mesh : tf.Tensor, new_func_shape : List[int] | None = None) -> 'Image':
        # mutates this image:
        # if we want to hold the mesh of a new function, 
        # but we have the same padded_mesh_shape 
        # and overall metadata
        self.mesh = new_mesh 

        if new_func_shape is not None:
            self.func_shape = new_func_shape 
            self.unpadded_slicer = padding_adjuster(self.pad, mesh_len=self.domain.dimension, func_len = len(self.func_shape))

        return self 
    
    def apply(self, f : Function, mutable = True) -> 'Image':
        # acts the given function on this image.
        # if in_place, acts directly onto self.mesh
        # if not, then it returns a new image with the function applied.

        X = self.view(padded=True, flattened=True)
        f = f(X)

        func_shape = tf.shape(f)[1:]
        
        # if the function is a scalar:
        # then we adjust func_shape to be []
        if len(func_shape) == 1 and func_shape[0] == 1:
            total_shape = self.padded_mesh_shape
            func_shape = []
        else:
            total_shape = tf.concat([self.padded_mesh_shape, func_shape], axis=0)
            func_shape = func_shape.numpy().tolist()

        func_mesh = tf.reshape(f, shape=total_shape)

        # if we can alter in-place, we do so.
        if mutable:
            return self._mutate(
                func_mesh,
                func_shape
            )
            
        else:
        # otherwise we return a new image.
            return Image(self.domain, func_mesh, total_shape, self.geometry, self.pad)

    


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

# I will be using the right handed base here.
PARTIAL_BASE = right_partial_base


def CreateKernel(base : KernelBase, dims, func_size, i) -> tf.Tensor:
    # this gives a differential kernel along the given axis
    # without dividing by the relevant dx.
    # so this takes in F and returns dF

    stencil, shape = base
    sizes = [1] * dims
    
    for j in shape:
        sizes[i] = j
        i += 1 
    
    stencil = tf.reshape(stencil, sizes + [1, 1])

    # reshape to [k1, k2, ..., kN, func_size, func_size]
    return tf.broadcast_to(stencil, sizes + [func_size, func_size])


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

    func_size = tf.reduce_prod(image.func_shape)

    if tf.size(image.func_shape) == 0:
        # expand out so our function out size becomes 1
        substrate = tf.expand_dims(image.mesh, axis=-1)
        # this will just be our default treatment of scalars - expand them to have dimension 1.
    
    else:
        # we compress our function output shape - we will later reshape this back
        reduction_shape = tf.concat([image.padded_mesh_shape, [func_size]], axis=0)
        substrate = tf.reshape(image.mesh, shape=reduction_shape)
    
    substrate = tf.expand_dims(substrate, axis=0)

    # we return the prepped substrate,
    # and the relevant dim_out for convolution also
    return (substrate, func_size)


# --- OPERATORS --- # 

# now we define operators.
# these act between images.
Operator = Callable[[Image], Image]


# this returns an operator that calculates the partial derivatives along the
# specified coordinates and stacks on the last axis. if empty, will calculate the full gradient.
def Partials(
    wrt = [] # the list of coordinate indices over which to compute partials
) -> Operator:

    # returns tensor of relevant partial derivatives of f over the given domain, 
    # same coordinate order as in wrt, stacked along last axis.
    def _h(phi : Image) -> Image:
        domain = phi.domain

        substrate, func_size = prep_for_kernel(phi)
        outputs = []

        # if axes are not specified, we calculate over all variables by default.
        for i in (wrt if wrt else range(domain.dimension)):
            kernel = CreateKernel(PARTIAL_BASE, domain.dimension, func_size, i)
            
            # we pad convolutions to the same size as before, and account for this by including
            # sufficient padding beforehand.
            post_kernel = tf.nn.convolution(substrate, kernel, padding='SAME')
            
            output = tf.squeeze(post_kernel, axis=0)
            output = tf.reshape(output, shape=phi.shape)

            outputs.append(output/domain.steps[i]) # divide by relevant dx
        
        # stack along the final axis
        final_mesh = tf.stack(outputs, axis=-1)
        final_shape = tf.shape(final_mesh)

        # ok. now we correct the padding of the geometry functions:
        # this operator reduces the padding by one, so we also adjust like so:
        dphi = Image(domain=domain, mesh=final_mesh, shape=final_shape, geometry=phi.geometry, pad=phi.pad)
        
        return dphi
    
    return _h 


# the full gradient.
# takes in an image of rank n,
# returns one of rank n+1
Gradient : Operator = Partials()


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