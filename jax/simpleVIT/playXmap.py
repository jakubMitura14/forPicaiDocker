#using xmap in a model https://github.com/venkat4938/flaxformer/blob/87cd1f420e6f686a595facab3a1328a4e16d53ff/flaxformer/architectures/perceiver_ar/slicing.py
# 3d mnist https://www.kaggle.com/datasets/daavoo/3d-mnist
import os
import jax
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
import jax.numpy as jnp
from jax import lax
from jax.nn import one_hot, relu
from jax.scipy.special import logsumexp
from jax.experimental.maps import Mesh
import jax
import numpy as np
from typing import Any, Callable
import flax.linen as nn
from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow_datasets as tfds
from jax.experimental.maps import xmap


w1 = jnp.zeros((784, 512))
w2 = jnp.zeros((512, 10))
images = jnp.zeros((128, 784,128))
labels = jnp.zeros(128, dtype=jnp.int32)

"""

"""
def named_predict(w1, w2, image):
  hidden = relu(lax.pdot(image, w1, 'inputs'))
  logits = lax.pdot(hidden, w2, 'hidden')
  return logits - logsumexp(logits, 'classes')

def named_loss(w12, images, labels):
    w1,w2=w12
    predictions = named_predict(w1, w2, images)
    num_classes = lax.psum(1, 'classes')
    bb=lax.psum(jnp.ravel(w1),'inputs')

    #   print(f"num_classes {num_classes}")
    targets = one_hot(labels, num_classes, axis='classes')
    losses = lax.psum(targets * predictions, 'classes')
    return -lax.pmean(losses,('batch'))#+bb


in_axes = [[{0:'inputs', 1:'hidden'},
           {0:'hidden', 1:'classes'}],
           {0:'batch', 1:'inputs'},
           {0:'batch'}]

loss = xmap(named_loss, in_axes=in_axes, out_axes=[...])
print(loss([w1, w2], images, labels))#should be 2.3


# bb=lax.concatenate(w1,0)


# # devices = np.array(jax.local_devices())
# # with Mesh(devices, ('x',)):
# #   print(loss(w1, w2, images, labels))
# bb=lax.concatenate(jnp.zeros((5, 5, 5, 5,5), dtype=jnp.float32),'inputs')
# bb.shape

# img = jnp.zeros((1,5, 200, 198, 126), dtype=jnp.float32)
# kernel = jnp.zeros((5, 5, 5, 5,5), dtype=jnp.float32)

# convolved = jax.lax.conv_general_dilated(img, kernel, (1, 1,1), 'SAME')
# jnp.ravel(convolved).shape

# from jax import lax
# import jax.numpy as jnp
# img = jnp.zeros((1, 200, 198, 3), dtype=jnp.float32)
# for k in range(3):
#   x = 30 + 60*k
#   y = 20 + 60*k
#   img = img.at[0, x:x+10, y:y+10, k].set(1.0)
# kernel = jnp.zeros((3, 3, 3, 3,3), dtype=jnp.float32)
# kernel += jnp.array([ [[1, 1, 0],
#                      [1, 0,-1],
#                      [0,-1,-1]],[[1, 1, 0],
#                      [1, 0,-1],
#                      [0,-1,-1]] ,[[1, 1, 0],
#                      [1, 0,-1],
#                      [0,-1,-1]]   ] )[:, :, jnp.newaxis, jnp.newaxis]

# out = lax.conv(img,    # lhs = NCHW image tensor
#                kernel, # rhs = OIHW conv kernel tensor
#                (1, 1,1),  # window strides
#                'SAME') # padding mode
# print("out shape: ", out.shape)


# # out = lax.conv_with_general_padding(
# #   jnp.transpose(img,[0,3,1,2]),    # lhs = NCHW image tensor
# #   jnp.transpose(kernel,[2,3,0,1]), # rhs = IOHW conv kernel tensor
# #   (1, 1),  # window strides
# #   ((2,2),(2,2)), # general padding 2x2
# #   (1,1),  # lhs/image dilation
# #   (1,1))  # rhs/kernel dilation



# # from jax.experimental.maps import xmap
# # from typing import Any, Callable
# # import jax.numpy as jnp
# # from jax import lax
# # from jax.nn import one_hot, relu
# # from jax.scipy.special import logsumexp
# # from jax.experimental.maps import Mesh
# # import jax
# # import numpy as np
# # from typing import Any, Callable
# # class ArrayType:
# #   def __getitem__(self, idx):
# #     return Any
# # f32 = ArrayType()
# # i32 = ArrayType()

# # def my_func(x: f32[{'main':5,'batch': 20}]) -> f32[{'main':5,'batch': 20}]:
# # #   assert x.shape == (5,)
# #   # assert x.named_shape == {'batch': 20}  # TODO: Implement named_shape
# #   return x

# # x: f32[(20, 5)] = jnp.zeros((20, 5), dtype=np.float32)
# # f = xmap(my_func,
# #          in_axes={0: 'batch'},   # Name the first axis of the only argument 'batch'
# #          out_axes={1: 'batch'})  # Place the 'batch' named axis of the output as the second positional axis
# # y: f32[(5, 20)] = f(x)
# # (y == x.T).all()


# # assert (y == x.T).all()  # The first dimension was removed from x and then re-inserted as the last dim




# # class CNN(nn.Module):
# #   """A simple CNN model."""

# #   @nn.compact
# #   def __call__(self, x):
# #     x = nn.Dense(features=10)(x)
# #     return x
# # cnn = CNN()
# # rng = jax.random.PRNGKey(0)
# # params = cnn.init(rng, jnp.ones(10))['params']

# # cnn.apply({'params': params},jnp.zeros((10)) )


