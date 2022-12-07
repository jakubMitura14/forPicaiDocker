

import flax.linen as nn
import jax.numpy as jnp

class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array.
  so in this case as far as I get it compat does nothing at all
  """
  @nn.compact
  def __call__(self, x):
    return x