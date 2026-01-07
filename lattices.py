# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Grids


from context import *
from typing import List
from functions import Function
from geometry import Geometry
from dataclasses import dataclass, field


# we discretize.
#
# the input is X [2, N] (the ranges) and dX [N] (the step sizes)
#
# where N is the dimension count, the first col tells us starting points and 
# second col tells us ending points.
#
# dX tells us how far away points should be in each dimension.
#
# so this will give a [B, n**N, N] grid, depending on the 'batches' parameter
#
# for now this splits our ranges linearly via tf.linspace
# but this is slightly suspect for more nontrivial geometries, like spherical
# still, though, should work not too bad.
#
# we use meshgrid to produce a cartesian product of linspaces
# as such I will make the indexing adjustable
def discretize(ranges : tf.Tensor, steps : tf.Tensor, padding=0, batches=1, indexing='ij') -> tf.Tensor: 

    if padding == 0:
        X = ranges 
    else:
        padder = steps * padding
        # pad evenly all around
        padding = tf.stack([-padder, padder], axis=0)
        X = ranges + padding
    
    dX = steps

    grid_slices = []

    starts = X[0]
    ends = X[1]

    N = tf.shape(starts)[0]

    for i in range(N):
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
    grid = tf.stack(grid, axis=-1)

    if batches:
        # we add the batch axis out front
        grid = tf.expand_dims(grid, axis=0)

        if batches > 1:
            # broadcast if necessary
            grid = tf.broadcast_to(grid, shape = [batches] + [1 for _ in range(N+1)] )


    return grid


# this is a helper function - it is a list of slices that throws away the 
# specified amount of padding.
# it is assumed that the first axis is always a batch axis.
def padding_adjuster(
        padding, # the amount of padding to "throw away" 

        dimension, # the amount of axes to cut
        # the leading axes are assumed to be the lattice axes

        rank = 0, # the amount of axes not to cut (assumed scalar function, len = 0)
        # the ending axes are assumed to be the function axes

        # if we take the grid to look like [B, n1, n2, ..., nN, f1, f2, ... fF]
        # where n1, n2 are the lattice axes and f1, f2... are the component axes of 
        # the function evaluated at the corresponding lattice points,
        # 
        # then dimension = N
        # and rank = F
    ) -> List[slice]:

    if padding == 0:
        # if the padding is 0, we just return [:,:,:,...]
        return [slice(None) for _ in range(1+dimension+rank)]
    
    # we return [:, pad:-pad, pad:-pad, ..., pad:-pad, :, :, ..., :]
    pad_slice = slice(padding, -padding)
    
    if rank == 0:
        return [slice(None)] + [pad_slice for _ in range(dimension)]
    
    return [slice(None)] + [pad_slice for _ in range(dimension)] + [slice(None) for _ in range(rank)]
    

# --- LATTICEES --- #

# this is a parent dataclass to wrap tensors that represent discretized domains/images in our framework.
# it will include general lattice operations that allow us to efficiently handle padding, flattening,
# and other such operations on our tensor lattice.
@dataclass
class Lattice:
    lattice_shape : Shape # the (UNPADDED) lattice shape - i.e. the leading (ignoring B) dimensions of @param grid that represent the 'lattice' axes which we will use to navigate the lattice 
    entry_shape : Sequence[int] # the entry shape - i.e. the ending dimensions of @param grid that represent the 'component' axes of some value evaluated at the corresponding lattice point, must SPECIFICALLY be an unpackable python structure of integers

    dimension : int  # the length of the lattice shape. equivalently, the amount of coordinate variables N
    rank : int  # the length of the entry shape. effectively, the tensor rank of our values.

    leading_shape : Shape # the total shape of the PADDED lattice along with the initial batch dimension
    shape : Shape # the total shape of the PADDED grid

    grid : tf.Tensor | None = None, # the grid we are wrapping
    # in general, the parameter should look like [(B), *M, *V],
    # where B is the (optional) batch axis,
    # M = [m1, m2, ..., mN] is the lattice shape,
    # and V = [v1, v2, ..., vM] is the entry shape
    #
    # patch -- previously grid was at the top of the attributes and did not allow None, 
    # i have now allowed None here so that we can treat "lattice generators" similarly to regular lattices.
    # if grid is None, it is expected that the view() method will be somehow overridden to generate a lattice to return. 
    # the reason for this is that holding DOMAINs statically throughout training has proven far more memory-intensive 
    # (especially for my 6GB gpu) than generating them from scratch each time we want to view them, and then deleting 
    # to free up memory for gradients. so I will include the choice to be dynamic with the creation of large lattices.

    padding : int = 0 # desired amount of padding.
    batches : int = 1 # B, the amount of batches in grid.

    unpadded_slicer : List[slice] = field(init=False) # allows us to slice away the padding automatically

    # we calculate the unpadded slicer after we instantiate
    def __post_init__(self):
        self.unpadded_slicer = padding_adjuster(self.padding, self.dimension, self.rank)
            
    # we view the entire grid as a tensor.
    # so this returns [B, n1, n2, ..., nN, *entry_shape]
    def view(self, padded=False, flattened=False) -> tf.Tensor:
        assert self.grid is not None, 'The view() method must be overridden for lattice generators.'

        # this is good if we have padding - we can view with/without padding at will.
        grid = self.grid if padded else self.grid[self.unpadded_slicer]

        # we can also flatten if needed, such as for function application
        new_shape = [-1, *self.entry_shape] if self.rank else [-1]
        return tf.reshape(grid, shape=new_shape) if flattened else grid

    # we view the i'th batch entry.
    # so this returns [n1, n2, ..., nN, *entry_shape]
    def get(self, i=0, padded=False, flattened=False) -> tf.Tensor:
        return self.view(padded=padded, flattened=flattened)[i]


# --- DOMAINS --- #

# this is a wrapped tensor dataclass that can discretize a given range of values
# TODO: generalize to other discretization methods. right now it just makes a linear mesh
class Domain(Lattice):
    def __init__(
        self,
        geometry : Geometry, # a pair of functions, each of which return [B, N, N]
        ranges : tf.Tensor, # should be a [2, N] tensor,
        steps : tf.Tensor, # should be a [N] tensor,  

        timed : bool = False, # we mark whether or not the first non-batch axis is temporal.  
        padding : int = 0, # we allow padding around the boundary for convolution purposes
        batches : int = 1, # how many batches we want. by default, 1 is a good choice, but we may want to explicitly broadcast multiple copies of our coordinate grid.
    
        dynamic : bool = False # we include the choice to not store the entire lattice statically, but recreate it dynamically per viewing. useful if memory is tight
    ):
        

        # this is the number N
        dimension = int(tf.shape(ranges)[1].numpy())
        
        # we create the coordinate grid.
        # this is a tensor of the shape [B, n1, n2, ..., nN, N] 
        # where every entry in the lattice is a 1-tensor corresponding to its own coordinates.
        coordinates = discretize(ranges=ranges, steps=steps, padding=padding, batches=batches)

        # we store shapes.

        # the unpadded shape OF THE LATTICE itself (ignoring the final axis of the coordinate grid)
        lattice_shape = tf.cast((ranges[1] - ranges[0])/steps, tf.int32)
        
        shape = tf.shape(coordinates)
        # the padded shape -- just measure the grid
        # we include the batch dimension and the lattice dimensions
        leading_shape = shape[:(dimension + 1)]

        # each entry is a 1-tensor representing a tuple of coordinates
        entry_shape = [dimension]
        rank = 1

        if dynamic:
            coordinates = None 
    
        super().__init__(grid=coordinates, lattice_shape=lattice_shape, entry_shape=entry_shape, dimension=dimension, rank=rank, leading_shape=leading_shape, shape=shape, padding=padding, batches=batches)
        
        # true if we include a temporal coordinate
        self.timed = timed 

        # the ranges of our axes: such as, x in [0.0, 5.0]
        self.ranges = ranges 
        # and the discrete steps between points, like, x in [0.0, 5.0] with dx=0.001
        self.steps = steps

        if dynamic:
            def view(padded=False, flattened=False) -> tf.Tensor:
                grid = discretize(ranges=ranges, steps=steps, padding=padding if padded else 0, batches=batches)
                return tf.reshape(grid, shape=[-1, dimension]) if flattened else grid
            
            self.view = view
        
        # we store the metric functions
        self.geometry = geometry



# --- IMAGES --- #

# these represent the "images" of certain functions or distributions over domains.
# as such, they hold a reference to their original domain for bookkeeping and retrieval purposes.
class Image(Lattice):
    def __init__(
        self,
        domain : Domain, # the domain

        # we trust in the user that the grid passed here will be compatible with the domain. i.e. the leading dimensions [B, n1, n2, ..., nN] will be equal to those of the coordinate grid
        grid : tf.Tensor, # the grid that we wrap here, a tensor of the shape [B, n1, n2, ..., nN, *V] where V is the shape of the lattice value at (x1, x2, ..., xN)
        
        shape : Shape | None = None,
        leading_shape : Shape = None,
        entry_shape : Sequence[int] | None = None,

        batches : int = 1 # how many batches we have (per batch in the domain). In Images, this tells us how many batches PER DOMAIN BATCH. so the total amount of batches B is self.batches * self.domain.batches
    ):
        
        self.domain = domain
        
        if shape is None:
            shape = tf.shape(grid) 
        
        if entry_shape is None:
            entry_shape = shape[(1 + domain.dimension):].numpy().tolist() # we keep this as a list for manipulability, this is F from above
        
        if leading_shape is None:
            leading_shape = shape[:(1+domain.dimension)]
            
        rank = len(entry_shape)

        super().__init__(
            grid=grid, 
            lattice_shape=domain.lattice_shape, 
            entry_shape=entry_shape, 
            dimension=domain.dimension, 
            rank=rank, 
            leading_shape=leading_shape, 
            shape=shape, 
            padding=domain.padding,
            batches=batches
        )

        '''
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
        
        '''


    
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
def Integral(image : Image, average=False, per_batch=False) -> tf.Tensor: 
    flattened = image.view(flattened=True) 

    '''
    # TODO: fix this (define geometry for images)
    g = image.geometry[0][padding_adjuster(image.padding, image.domain.dimension, 2)]  # we get the unpadded metric
    flat_sqrt_g = tf.reshape(tf.sqrt(tf.abs(tf.linalg.det(g))), shape=[-1]) # we find the sqrt of det g

    dVol = flat_sqrt_g * tf.reduce_prod(image.domain.steps) # we multiply by dXdYdZ...

    new_shape = tf.concat([tf.shape(dVol), [1 for _ in image.func_shape]], axis=0)
    
    dVol = tf.reshape(dVol, new_shape)

    integrable = flattened * dVol
    '''
    integrable = flattened

    if per_batch:
        # if we are doing this per batch, we want to keep the batch dimension and sum
        # only over the lattice
        batch_dims = image.batches * image.domain.batches
        shape = [batch_dims, -1, *image.entry_shape] if image.rank else [batch_dims, -1]
        integrable = tf.reshape(integrable, shape=shape)

        reduction_axis = 1
        # in this case we return [B, *entry_shape]
    
    else:
        # in this case we return [*entry_shape]
        reduction_axis = 0

    # if average, we return the mean over the lattice
    return tf.reduce_mean(integrable, axis=reduction_axis)  if average else tf.reduce_sum(integrable, axis=reduction_axis)
