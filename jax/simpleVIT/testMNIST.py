
from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow_datasets as tfds


from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf
from absl import app
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




try:
    app.run(lambda argv: None)
except:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', '/workspaces/forPicaiDocker/jax/data', 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    '/workspaces/forPicaiDocker/jax/configs/default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=True)

configCurr = ml_collections.ConfigDict()
configCurr.patches=ml_collections.ConfigDict({'size': (16, 16)})
configCurr.transformer = ml_collections.ConfigDict()
configCurr.hidden_size = 192
configCurr.transformer.mlp_dim = 10
configCurr.transformer.num_heads = 2
configCurr.transformer.num_layers = 1
# configCurr.learning_rate=0.01


num_classes=10
model = VisionTransformer(num_classes=num_classes, **configCurr)



@jax.jit
def apply_model(state, images, labels,rng):
  _, new_rng = jax.random.split(rng)
  dropout_rng = new_rng#jax.random.fold_in(rng, jax.lax.axis_index(0))
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, images,rngs=dict(dropout=dropout_rng),train=True)
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_accuracy = []

  for perm in perms:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]
    grads, loss, accuracy = apply_model(state, batch_images, batch_labels,rng)
    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy


config = ml_collections.ConfigDict()
config.patches=ml_collections.ConfigDict({'size': (16, 16)})
config.transformer = ml_collections.ConfigDict()
config.hidden_size = 192


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds


def create_train_state(rng, config):
  """Creates initial `TrainState`."""
  cnn = VisionTransformer(num_classes=num_classes, **config)
  params = cnn.init(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension for initialization.
        jnp.ones([1, 16, 16, 1], jnp.float32),
        train=False)['params']
  learning_rate=0.01
  momentum=0.9

  tx = optax.sgd(learning_rate, momentum)
  
  return train_state.TrainState.create(
      apply_fn=cnn.apply, params=params, tx=tx)


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
  """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    The train state (which includes the `.params`).
  """
  train_ds, test_ds = get_datasets()
  rng = jax.random.PRNGKey(0)
  batch_size=5
  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(dict(config))

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, config)
  num_epochs=5
  for epoch in range(1, num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                    batch_size,
                                                    input_rng)
    _, test_loss, test_accuracy = apply_model(state, test_ds['image'],
                                              test_ds['label'],rng)

    logging.info(
        'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
        % (epoch, train_loss, train_accuracy * 100, test_loss,
           test_accuracy * 100))

    summary_writer.scalar('train_loss', train_loss, epoch)
    summary_writer.scalar('train_accuracy', train_accuracy, epoch)
    summary_writer.scalar('test_loss', test_loss, epoch)
    summary_writer.scalar('test_accuracy', test_accuracy, epoch)

  summary_writer.flush()
  return state






def main():

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  train_and_evaluate(configCurr, FLAGS.workdir)


flags.mark_flags_as_required(['config', 'workdir'])
main()