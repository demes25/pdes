from operators import *

def sq(v):
    return v*v

div = grad(sq)

print(div(tf.constant([0, 2, 3, 4], dtype=tf.float32)))