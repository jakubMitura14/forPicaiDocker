# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Sequence, TypeVar

from flax import linen as nn
import jax.numpy as jnp
from jax import lax
from jax.nn import one_hot, relu
from jax.scipy.special import logsumexp

T = TypeVar('T')

def weight_standardize(w, axis, eps,forMean_axes):
  """
  So simple standardize
  Subtracts mean and divides by standard deviation."""
  w = w - lax.pmean(w,forMean_axes)
  w = w / (jnp.std(w, axis=axis) + eps)
  return w


class StdConv(nn.Conv):
  """
  Convolution with weight standardization.
  ok so we extend jax Conv by defining parameters diffrently so we as far as I see just standardize the numbers along the kernel  
  """
  def param(self,
            name: str,
            init_fn: Callable[..., T],
            *init_args) -> T:
    param = super().param(name, init_fn, *init_args)
    if name == 'kernel':
      param = weight_standardize(param, axis=[0, 1, 2], eps=1e-5,)# TODO( add axis 3)
    return param


class ResidualUnit(nn.Module):
  """
  Bottleneck ResNet block.
  
  """
  features: int
  strides: Sequence[int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    #first we check weather the shape of input and output agrees if I get it right
    needs_projection = (
        x.shape[-1] != self.features * 4 or self.strides != (1, 1))#TODO strides should be 3 dim

    residual = x
    if needs_projection:
      # we see that in case of projection we set 4 times more features and strides 1,1
      #so here we basically adapt residual if stride is not 1,1 so none; also we check feature channel as far as I can see 
      # we have here the numpy convention so I suppose batch is last dimension and channels/ features is almost last
      # it seems also it is assumed that true number of channels is 4 times those of features
      residual = StdConv(
          features=self.features * 4,
          kernel_size=(1, 1), 
          strides=self.strides,
          use_bias=False,
          name='conv_proj')(
              residual)
      residual = nn.GroupNorm(name='gn_proj')(residual)
    # couple convolutions with added standarization and group normalization
    y = StdConv(
        features=self.features,
        kernel_size=(1, 1),
        use_bias=False,
        name='conv1')(
            x)
    y = nn.GroupNorm(name='gn1')(y)
    y = nn.relu(y)
    y = StdConv(
        features=self.features,
        kernel_size=(3, 3),
        strides=self.strides,
        use_bias=False,
        name='conv2')(
            y)
    y = nn.GroupNorm(name='gn2')(y)
    y = nn.relu(y)
    y = StdConv(
        features=self.features * 4,
        kernel_size=(1, 1),
        use_bias=False,
        name='conv3')(
            y)

    y = nn.GroupNorm(name='gn3', scale_init=nn.initializers.zeros)(y)
    # below main idea of residual block so we add the output of those convolutions to the input
    y = nn.relu(residual + y)
    return y


class ResNetStage(nn.Module):
  """
  as explained in https://www.youtube.com/watch?v=wOuaGvxbtZo
  Resnet basicall is to avoid problems with diminishing and exploding gradients 
  So idea is to make sth like a lower bound so when we add more layers they should be no worse than identity mapping
  as identity mapping will not reduce the performance - now when we have lower bound we can than improve over this so we have some layers and we sum the output of those layers via skip connection with the input to those layers

  A ResNet stage.
  """
  block_size: Sequence[int]
  nout: int
  first_stride: Sequence[int]

  #compact enable some inlining  and below we just collect multiple res net blocks 
  @nn.compact
  def __call__(self, x):
    x = ResidualUnit(self.nout, strides=self.first_stride, name='unit1')(x)
    for i in range(1, self.block_size):
      x = ResidualUnit(self.nout, strides=(1, 1), name=f'unit{i + 1}')(x)
    return x
