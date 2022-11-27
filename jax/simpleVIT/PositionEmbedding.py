
from typing import Any, Callable, Optional, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer.
  """

  posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

  @nn.compact
  def __call__(self, inputs):
    """Applies the AddPositionEmbs module.
    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,' 
                              ' but it is: %d' % inputs.ndim) #TODO(adapt to 3 dim)
    
    # here it seems simplest possible idea so we just add some array to be learned that has good shape so why it is called positional embedding?
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2]) #TODO(adapt to 3 dim)
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
    return inputs + pe