# Raveena D'Souza
# Feb 13, 2018
# Using Google's TensorFlow machine learning library for the visualisation of the mandelbrot set

# import needed libraries
import tensorflow as tf
import numpy as np

# visualisation imports
import PIL.Image
from io import BytesIO
from IPython.display import Image, display

def displayFractal(a, frmt='jpeg'):
    # an array of iteration counts is displayed as a fractal
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    image = np.concatenate([10+20*np.cos(a_cyclic),
                            30+50*np.sin(a_cyclic),
                            155-80*np.cos(a_cyclic)], 2)
    image[a == a.max()] = 0
    a = image
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, frmt)
    display(Image(data=f.getvalue()))

# create a session to run
sess = tf.InteractiveSession()

# create a 2D array of complex numbers with NumPy
y, x = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
z= x+1j*y

# define and initialise tensors
xs = tf.constant(z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))

# explicitly initialise variables before usage
tf.global_variables_initializer().run()

# computation of z: z^2 + x
zs_ = zs*zs + xs

# check if diverged
not_diverged = tf.abs(zs_) < 4

# update the zs and iteration count
step = tf.group(
  zs.assign(zs_),
  ns.assign_add(tf.cast(not_diverged, tf.float32))
  )

for i in range(200): step.run()

# finally, display the fractal
displayFractal(ns.eval())
