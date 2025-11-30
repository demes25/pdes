A convolutional neural network to solve inhomogeneous partial differential equations on discretized domains.

This project uses TensorFlow and TensorFlow-Plot. 

I began this project as a concrete exercise and demonstration in the utility of neural networks in dynamics problems 
in physics. 
It is coupled with a research project on Bumblebee fields in a Schwarzschild background. A primary objective at this point
is to apply this system to find discretized solutions to the relevant differential equations in this research. As such,
a primary programming objective at this time is to effectively handle domains in the Schwarzschild geometry while maintaining
the simplicity of TensorFlow's vectorized discretization algorithms.


# CONTENTS
A library that allows the construction of spatial partial differential equations on a discretized mesh. 
I have here defined some modules that may be of use for other general applications.

Ideally, this program will include an extensive library to deal with different coordinate systems and geometries.
In its current iteration, it works for (and has differential operators defined for) flat cartesian coordinates,
though it also includes early frameworks for handling different metrics and geometries.

# classes:
- Domain - a storage type which holds the coordinate ranges, discretization step sizes, and the relevant Geometry (see typedefs). 
- Image - a type which holds the mesh corresponding to the values of some function evaluated at each discrete point in a given Domain. it also includes the metric tensors evaluated at each point in the discretized domain. by default, we may take the image directly of a domain, which in this case simply creates the "coordinate image" - a mesh in which every point simply holds a rank-1 tensorflow tensor encoding its own coordinates.  
# typedefs: 
- Function - a callable that returns a tensorflow tensor. It is expected that functions will act on tensors I : [B, input_shape...] and return O : [B, output_shape...].
- Geometry - a tuple of two Functions which return the metric and inverse metric, respectively, when given some batched coordinate points X : [B, N] where we have N coordinate dimensions.
- KernelBase - a tuple of containing a tensorflow tensor and a list of ints. It encodes the 'base' of some differential kernel along with its shape (the list of ints), which must then be reshaped and broadcasted according to what it is being applied to. 
- Operator - a callable that acts on an Image and returns an Image. It is mostly used to type-hint differential operators (created from a KernelBase) which discretely differentiate image meshes.
- Distribution - also a callable that acts on an Image and returns an Image, however this is used in a different context than Operator. While Operator is usually some kind of convolution kernel, this is rather a callable that expects some coordinate image, acts a Function on those coordinates, and returns the resultant image. Right now, I have a couple distributions already defined, namely the normalized Gaussian G(mean, variance) and the reciprocal function 1/(|x| + epsilon).
# callables:
- Integral - acts on an Image, with some args, and returns a tensorflow tensor. It uses the relevant metrics to calculate the discretized integral (Riemann sum) of the image mesh. It can also return an average (divide by total amount of points in the mesh).
