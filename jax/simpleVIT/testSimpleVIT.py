import models_vit
from models_vit import VisionTransformer

import functools
import os
import time

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax.training import checkpoints as flax_checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

config = ml_collections.ConfigDict()
config.patches=ml_collections.ConfigDict({'size': (16, 16)})
config.transformer = ml_collections.ConfigDict()
config.hidden_size = 192
num_classes=2
model = VisionTransformer(num_classes=num_classes, **config)
train_state=[]
def cross_entropy_loss(*, logits, labels):
  labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
  return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()

def compute_metrics(*, logits, labels):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics

def create_train_state(rng, learning_rate, momentum,xDim,yDim):
  """Creates initial `TrainState`."""
  params = model.init(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension for initialization.
        jnp.ones((xDim,yDim), jnp.float32),
        train=False)['params']

  tx = optax.sgd(learning_rate, momentum)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits = VisionTransformer(num_classes=num_classes, **config).apply({'params': params}, batch['image'])
    loss = cross_entropy_loss(logits=logits, labels=batch['label'])
    return loss, logits
  grad_fn = jax.grad(loss_fn, has_aux=True)
  grads, logits = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=batch['label'])
  return state, metrics


@jax.jit
def eval_step(params, batch):
  logits = VisionTransformer(num_classes=num_classes, **config).apply({'params': params}, batch['image'])
  return compute_metrics(logits=logits, labels=batch['label'])


def train_epoch(state, train_ds, batch_size, epoch, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state, metrics = train_step(state, batch)
    batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}

    print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
        epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

    return state



def get_datasets():
  """Load MNIST train and test datasets into memory."""
  
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds

print(model)







# batch = next(iter(ds_train))

# #as we see below batch is a dict 
# def init_model():
    # return model.init(
    #     jax.random.PRNGKey(0),
    #     # Discard the "num_local_devices" dimension for initialization.
    #     jnp.ones(batch['image'].shape[1:], batch['image'].dtype.name),
    #     train=False)



# ds_train, ds_test = input_pipeline.get_datasets(config)
# #here we get as far as I get numpy into float32 jax
#   train_ds['image'] = jnp.float32(train_ds['image']) / 255.
#   test_ds['image'] = jnp.float32(test_ds['image']) / 255.




# # Use JIT to make sure params reside in CPU memory.
# variables = jax.jit(init_model, backend='cpu')()
# total_steps = 10

# lr_fn = utils.create_learning_rate_schedule(total_steps, config.base_lr,
#                                             config.decay_type,
#                                             config.warmup_steps)

# tx = optax.chain(
#     optax.clip_by_global_norm(config.grad_norm_clip),
#     optax.sgd(
#         learning_rate=lr_fn,
#         momentum=0.9,
#         accumulator_dtype='bfloat16',
#     ),
# )
# initial_step = 1
# opt_state = tx.init(params)


# def cross_entropy_loss(*, logits, labels):
#   labels_onehot = jax.nn.one_hot(labels, num_classes=10)
#   return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


# def compute_metrics(*, logits, labels):
#   loss = cross_entropy_loss(logits=logits, labels=labels)
#   accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
#   metrics = {
#       'loss': loss,
#       'accuracy': accuracy,
#   }
#   return metrics