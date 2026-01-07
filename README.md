# PDEs
Convolutional neural networks (CNN) and trainable meshes for solving eigenvalue and inverse problems for partial differential equations on discretized domains.

This project uses TensorFlow [https://www.tensorflow.org/] and TensorFlow-Plot [https://pypi.org/project/tensorflow-plot/]. 

## Overview 

I began this project as a concrete exercise and demonstration in the utility of neural networks (NNs) in dynamics problems 
in physics. 
It is coupled with a research project on Bumblebee fields in a Schwarzschild background. A primary objective at this point
is to apply this system to find discretized solutions to the relevant differential equations in said research project. As such, a programming objective at this time is to effectively handle domains in the Schwarzschild geometry while maintaining the simplicity of TensorFlow's vectorized discretization algorithms.

For the neural network architecture, I initially tried a simple n-layer MLP on homogeneous eigenvalue problems (to no avail). Additionally, I was trying to apply this MLP onto the coordinate mesh, which was not the most mathematically sound idea (for example - the solution to Laplace(U) = 0 should depend purely on our domain and our boundary conditions, and not on our choice of coordinates).
I then remembered that many linear inhomogeneous PDEs rely on Green's functions or 'solution kernels' which we may convolve with the forcing term to yield a particular solution. The NN analogues of such convolutional solution kernels are, naturally, convolutional NNs. After some digging, I found some good CNN architectures that work well in solving PDEs and acting as Physics-informed NNs.

All that to say:
This program currently uses the U-Net convolutional neural network architecture for inverse problems, which is detailed at length at https://www.sciencedirect.com/topics/computer-science/u-net 

For the eigenvalue problems (in-dev), I will be using trainable meshes (tensor-valued tf.Variables) along with boundary values and constraints. 

Tests, logs, images, and relevant notes are stored in the 'logs' directory.

More hints and comments are available in the code files.

## Contents

A library that allows the construction of spatial partial differential equations on a discretized mesh. 
This project includes modules that may be (hopefully *will* be) of use for other general applications.

Ideally, this program will include an extensive library to deal with different coordinate systems and geometries.
In its current iteration, it works for (and has differential operators defined for) flat cartesian coordinates,
though it also includes early frameworks for handling different metrics and geometries (to be sophisticated).

### Classes

#### Domain 
A class which encodes a "domain" - i.e. a collection of points. This type holds the associated coordinate ranges, discretization step sizes, and Geometry (see #typedefs). It also contains the coordinate lattice, which is a discretized "mesh" spanning the specified coordinate ranges with the given discretization settings.

#### Image
A class which holds the lattice corresponding to the values of some function evaluated at each discrete point in a given Domain.  

#### System 
A class which holds an operator L, as well as a forcing term f and/or a bunch of constraints and boundary conditions that represents our partial differential equation system (currently only for inverse problems). It also contains a method train() to which we may pass a candidate solution network U and some other args, whence the system will train the given network on the PDE that the system represents. 
The main loss ("operator loss") is calculated by taking (some scaling of) the integral of p(E) over the entire mesh, where E is the "error term" representing L[U] - f, i.e. the difference between the reconstructed forcing term and the true forcing term, and where p is the "pointwise loss" function that we act on each point before we integrate (by default, it is just tf.square, so that the integrand is purely positive. tf.abs is a viable alternative). 
The total loss is a weighted combination of the operator loss and the boundary conditions. Note that the pointwise loss function p and the boundary conditions may be passed as constructor arguments for a System object and as such are left up to the wisdom of the user.

#### SolutionNetwork 
A keras Model subclass which acts as a superclass for our proposed solution networks. 
##### SpatialKernel 
A SolutionNetwork utilizing the U-Net architecture noted above. Instances of this class will act as our proposed solutions for inverse problems.

### Typedefs

#### Function
Any callable that returns a tf.Tensor. Usually the input will also be a tf.Tensor with some additional args. I have chosen the convention that all "irrelevant" input axes must be collapsed into the batch axis, with the caveat that if there are NO "irrelevant" axes, the batch axis must nonetheless be added. 
For example, if our function expects to act on a rank-2 tensor and we have a [B, n1, n2, ...] lattice of rank-2 tensors, we must first collapse the leading lattice axes into a single batch dimension. 

#### Functional
Any callable that returns a Function. These are factories for Functions that take external or geometry-dependent args, such as the norm-square function which requires a specified Geometry. 

#### Distribution 
A callable that acts a Function on a Domain and returns an Image. It expects a coordinate domain, acts a specified Function on those coordinates, and returns the resultant image. Distributions are expected to complete the "axis gymnastics" (noted above for Functions) under-the-hood so as to effectively and systematically apply Functions to Domains with arbitrary lattice dimensions. Right now, I have a couple distributions already pre-defined, such as the normalized Gaussian G(mean, variance) and the reciprocal function 1/(|x| + epsilon), among others.

#### Operator 
A callable that acts on an Image and returns an Image. It is mostly used to type-hint differential operators (created from a KernelBase) which discretely differentiate image meshes, but it generally refers to any Image-to-Image callable.

#### KernelBase 
A tuple of containing a tensorflow tensor and a list of ints. It encodes the 'base' of some differential kernel along with its shape (the list of ints), which must then be reshaped and broadcasted according to what it is being applied to. 

#### Geometry 
A tuple of two Functions which return the metric and inverse metric, respectively, when given some batched coordinate points X : [B, N] where we have N coordinate dimensions.

