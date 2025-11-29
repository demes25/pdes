# Demetre Seturidze
# NN Field Theories
# Nov 2025
# Functions/Distributions

import tensorflow as tf
from context import *
from typing import Callable

# here we define functions.
# 
# we assume that functions will act individually on points
# and therefore do not need to decode mesh structure like operators do.
# 
# we will assume that the input tensor for any function look slike [B, ...]
# where the mesh axes are flattened into the batch dimension.
Function = Callable[..., tf.Tensor]


def delta(X : tf.Tensor):
    
