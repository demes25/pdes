# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Systems

import tensorflow as tf

# here we look at general systems and how to deal with them:
# discretizing the domain, passing through neural networks, 
# applying lagrangians and boundary conditions


class System:
    
    # we discretize domains.
    #
    # the input is X [N, 2] (the ranges) and dX [N] (the step sizes)
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
    def discretize(X, dX, indexing='xy'): 

        grid_slices = []

        for i in range(tf.shape(X)[0]):
            start = X[i, 0]
            end = X[i, 1]
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

