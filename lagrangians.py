# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Lagrangians

from operators import *
from geometry import Norms, InnerProducts, MinkowskiGeometry

# here we define some classical lagrangians on certain geometries
# some will not require any geometry.

DefaultMetrics = MinkowskiGeometry()

# these will be functionals that take in hyperparameters
# and return functionals that take in a field and return the lagrangian density at the given field value 
def FreeScalarField(
    m = zero, # mass of the scalar field
    metrics = DefaultMetrics # metrics should be a tuple of tensor functions (g, g_inv)
):
    _, g_inv = metrics

    if m == zero:
        def _s(phi):
            kinetic_term = Norms(grad(phi), g_inv)

            return scale_fn(half, kinetic_term)

    else:
        def _s(phi):
            kinetic_term = Norms(grad(phi), g_inv) 
            mass_term = scale_fn(-m*m, mul_fn(phi, phi))

            return scale_fn(half, add_fn(kinetic_term, mass_term)) 

    return _s