A convolutional neural network (CNN) to solve inhomogeneous partial differential equations on discretized domains.

This project uses TensorFlow and TensorFlow-Plot. 

I began this project as a concrete exercise and demonstration in the utility of neural networks (NNs) in dynamics problems 
in physics. 
It is coupled with a research project on Bumblebee fields in a Schwarzschild background. A primary objective at this point
is to apply this system to find discretized solutions to the relevant differential equations in this research. As such,
a primary programming objective at this time is to effectively handle domains in the Schwarzschild geometry while maintaining
the simplicity of TensorFlow's vectorized discretization algorithms.

For the neural network architecture, I initially tried a simple n-layer MLP on homogeneous eigenvalue problems (to no avail). Additionally, I was trying to apply this MLP onto the coordinate mesh, which was not the most mathematically sound idea (for example - the solution to Laplace(U) = 0 should depend purely on our domain and our boundary conditions, and not on our choice of coordinates).
I then remembered that many linear inhomogeneous PDEs rely on Green's functions or 'solution kernels' which we may convolve with the forcing term to yield a particular solution. The NN analogues of such convolutional solution kernels are, naturally, convolutional NNs. After some digging, I found some good CNN architectures that work well in solving PDEs and acting as Physics-informed NNs.

All that to say:
This program currently uses the U-Net convolutional neural network architecture, which is detailed at length at 
# https://www.sciencedirect.com/topics/computer-science/u-net


# CONTENTS
A library that allows the construction of spatial partial differential equations on a discretized mesh. 
I have here defined some modules that may be of use for other general applications.

Ideally, this program will include an extensive library to deal with different coordinate systems and geometries.
In its current iteration, it works for (and has differential operators defined for) flat cartesian coordinates,
though it also includes early frameworks for handling different metrics and geometries.

# classes:
- Domain - a storage type which holds the coordinate ranges, discretization step sizes, and the relevant Geometry (see #typedefs). 
- Image - a type which holds the mesh corresponding to the values of some function evaluated at each discrete point in a given Domain. it also includes the metric tensors evaluated at each point in the discretized domain. by default, we may take the image directly of a domain, which in this case simply creates the "coordinate image" - a mesh in which every point simply holds a rank-1 tensorflow tensor encoding its own coordinates.  
- System - a type which holds an operator L, forcing term f, and a bunch of constraints and boundary conditions that represents our partial differential equation system. It also contains a method train() to which we may pass a candidate solution network U, a domain D, etc etc which will train the given network on the PDE that the system represents. The main loss ("operator loss") is calculated by taking the ("average," see #callables below) integral of p(E) over the entire mesh, where E is the "error term" representing L[U] - f, i.e. the difference between the reconstructed forcing term and the true forcing term, and where p is the "pointwise loss" that we act on each point before we integrate (by default, it is just tf.square, so that the integrand is purely positive). The total loss is a weighted combination of the operator loss and the boundary conditions. Note that the pointwise loss and the boundary conditions may be passed as constructor arguments for a System object and as so are left up to the wisdom of the user.
- SpatialKernel - a keras Model subclass built on the U-Net architecture noted above. This will be our proposed solution.

# typedefs: 
- Function - a callable that returns a tensorflow tensor. It is expected that functions will act on tensors I : [B, input_shape...] and return O : [B, output_shape...].
- Geometry - a tuple of two Functions which return the metric and inverse metric, respectively, when given some batched coordinate points X : [B, N] where we have N coordinate dimensions.
- KernelBase - a tuple of containing a tensorflow tensor and a list of ints. It encodes the 'base' of some differential kernel along with its shape (the list of ints), which must then be reshaped and broadcasted according to what it is being applied to. 
- Operator - a callable that acts on an Image and returns an Image. It is mostly used to type-hint differential operators (created from a KernelBase) which discretely differentiate image meshes.
- Distribution - also a callable that acts on an Image and returns an Image, however this is used in a different context than Operator. While Operator is usually some kind of convolution kernel, this is rather a callable that expects some coordinate image, acts a Function on those coordinates, and returns the resultant image. Right now, I have a couple distributions already defined, namely the normalized Gaussian G(mean, variance) and the reciprocal function 1/(|x| + epsilon).

# callables:
- Integral - acts on an Image, with some args, and returns a tensorflow tensor. It uses the relevant metrics to calculate the discretized integral (Riemann sum) of the image mesh. It can also return an average (divide by total amount of points in the mesh).
