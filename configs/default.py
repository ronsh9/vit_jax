# Copyright 2024 The Flax Authors.
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

# Copyright 2021 The Flax Authors.
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
"""Default Hyperparameter configuration."""

import ml_collections

# ViT config is taken from: https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
def get_b16_config_deit_recipe():
  """Returns the ViT-B_16 configuration with training hyperparameters according to the DeiT recipe."""

  config = get_config_deit_recipe()

  # Model
  config.model = 'ViT'

  # Model params
  config.model_name = 'ViT-B_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 768
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 3072
  config.transformer.num_heads = 12
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.transformer.droppath_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None

  return config

# ViT config is taken from: https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
def get_b16_config_mae_recipe():
  """Returns the ViT-B_16 configuration with training hyperparameters according to the MAE recipe."""

  config = get_config_mae_recipe()

  # Model
  config.model = 'ViT'

  # Model params
  config.model_name = 'ViT-B_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 768
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 3072
  config.transformer.num_heads = 12
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.transformer.droppath_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None

  return config

# ViT config is taken from: https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
def get_l16_config_mae_recipe():
  """Returns the ViT-L_16 configuration with training hyperparameters according to the MAE recipe."""

  config = get_config_mae_recipe()

  # Model
  config.model = 'ViT'

  # Model params
  config.model_name = 'ViT-L_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 1024
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 4096
  config.transformer.num_heads = 16
  config.transformer.num_layers = 24
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.1 # originally this should be 0.1
  config.transformer.droppath_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None
  return config


def get_config_deit_recipe():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Dataset
  config.dataset = dataset = ml_collections.ConfigDict()
  dataset.name = 'imagenet'
  dataset.root = '/kmh-nfs-us-mount/data/imagenet' # FLAG: overwritten in scripts/run_remote.sh
  dataset.num_workers = 4 # FLAG: overwritten in scripts/run_remote.sh
  dataset.prefetch_factor = 2 # FLAG: overwritten in scripts/run_remote.sh
  dataset.pin_memory = False
  dataset.cache = False # FLAG: overwritten in configs/tpu_vit_deit_recipe.py
  dataset.input_size = 224

  # For AdamW optimizer:
  config.adamw_b1 = 0.9
  config.adamw_b2 = 0.95
  config.ema_decay = None # update according to: ema = (1-decay)*t + decay*ema | 0. or None to disable

  # For SGD optimizer:
  config.momentum = 0.9

  # Training
  config.learning_rate = 0.0005 # FLAG: overwritten in scripts/run_remote.sh
  config.warmup_epochs = 5
  config.batch_size = 1024 # FLAG: overwritten in scripts/run_remote.sh
  config.weight_decay = 0.05 # FLAG: overwritten in scripts/run_remote.sh
  
  config.shuffle_buffer_size = 16 * 128
  config.prefetch = 10

  config.num_epochs = 300 # FLAG: overwritten in scripts/run_remote.sh
  config.log_per_step = 100 # FLAG: overwritten in scripts/run_remote.sh
  config.log_per_epoch = -1
  config.eval_per_epoch = 1
  config.checkpoint_per_epoch = 20

  config.steps_per_eval = -1

  config.half_precision = False # FLAG: overwritten in configs/tpu_vit_deit_recipe.py

  config.seed = 0  # init random seed

  # Data augmentation
  config.dataset.aug = ml_collections.ConfigDict()
  config.dataset.aug.autoaug = 'rand-m9-mstd0.5-inc1'
  config.dataset.aug.interpolation = 'bicubic'
  config.dataset.aug.scale = (0.08, 1.0)  # scale augmentation
  config.dataset.aug.ratio = (3. / 4., 4. / 3.)  # aspect ratio augmentation

  config.dataset.aug.label_smooth = 0.1 # FLAG: overwritten in scripts/run_remote.sh
  config.dataset.aug.mixup_alpha = 0.8  # 0. to disable | FLAG: overwritten in scripts/run_remote.sh
  config.dataset.aug.cutmix_alpha = 1.0  # 0. to disable | FLAG: overwritten in scripts/run_remote.sh

  config.dataset.aug.reprob = 0.25 # for erasing (probability of erasing)
  config.dataset.aug.remode = 'pixel' # for erasing (random erase mode)
  config.dataset.aug.recount = 1 # for erasing (random erase count)
  config.dataset.aug.color_jitter = 0.3

  config.dataset.aug.deit_augmentations = True
  config.dataset.aug.src = False # simple random crop

  return config


def get_config_mae_recipe():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Dataset
  config.dataset = dataset = ml_collections.ConfigDict()
  dataset.name = 'imagenet'
  dataset.root = '/kmh-nfs-us-mount/data/imagenet' # FLAG: overwritten in scripts/run_remote.sh
  dataset.num_workers = 4 # FLAG: overwritten in scripts/run_remote.sh
  dataset.prefetch_factor = 2 # FLAG: overwritten in scripts/run_remote.sh
  dataset.pin_memory = False
  dataset.cache = False # FLAG: overwritten in configs/tpu_vit_mae_recipe.py
  dataset.input_size = 224

  # For AdamW optimizer:
  config.adamw_b1 = 0.9
  config.adamw_b2 = 0.95
  config.ema_decay = 0.9999 # update according to: ema = (1-decay)*t + decay*ema | 0.to disable

  # For SGD optimizer:
  config.momentum = 0.9

  # Training
  config.learning_rate = 1e-4 # FLAG: overwritten in scripts/run_remote.sh
  config.warmup_epochs = 20
  config.batch_size = 4096 # FLAG: overwritten in scripts/run_remote.sh
  config.weight_decay = 0.3 # FLAG: overwritten in scripts/run_remote.sh
  
  config.shuffle_buffer_size = 16 * 128
  config.prefetch = 10

  config.num_epochs = 300 # FLAG: overwritten in scripts/run_remote.sh
  config.log_per_step = 100 # FLAG: overwritten in scripts/run_remote.sh
  config.log_per_epoch = -1
  config.eval_per_epoch = 1
  config.checkpoint_per_epoch = 20

  config.steps_per_eval = -1

  config.half_precision = False # FLAG: overwritten in configs/tpu_vit_deit_recipe.py

  config.seed = 0  # init random seed

  # Data augmentation
  config.dataset.aug = ml_collections.ConfigDict()
  config.dataset.aug.autoaug = 'rand-m9-mstd0.5-inc1'
  config.dataset.aug.interpolation = 'bicubic'
  config.dataset.aug.scale = (0.08, 1.0)  # scale augmentation
  config.dataset.aug.ratio = (3. / 4., 4. / 3.)  # aspect ratio augmentation

  config.dataset.aug.label_smooth = 0.1 # FLAG: overwritten in scripts/run_remote.sh
  config.dataset.aug.mixup_alpha = 0.8  # 0. to disable | FLAG: overwritten in scripts/run_remote.sh
  config.dataset.aug.cutmix_alpha = 1.0  # 0. to disable | FLAG: overwritten in scripts/run_remote.sh

  config.dataset.aug.reprob = 0.25 # for erasing (probability of erasing)
  config.dataset.aug.remode = 'pixel' # for erasing (random erase mode)
  config.dataset.aug.recount = 1 # for erasing (random erase count)
  config.dataset.aug.color_jitter = 0.0

  config.dataset.aug.deit_augmentations = False
  config.dataset.aug.src = False # simple random crop

  return config


def metrics():
  return [
      'train_loss',
      'eval_loss',
      'train_accuracy',
      'eval_accuracy',
      'steps_per_second',
      'train_learning_rate',
  ]
