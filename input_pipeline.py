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

"""ImageNet input pipeline."""

import numpy as np
import os
import random
import jax
import torch
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from absl import logging
from functools import partial

import PIL
import timm
from timm.data import create_transform
from timm.data.mixup import Mixup


IMAGE_SIZE = 224  # TODO{km}: move to config
CROP_PADDING = 32  # TODO{km}: move to config
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


class ImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform, loader):
      super().__init__(root=root, transform=transform, loader=loader)
      self.num_classes = len(self.classes)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        onehot = np.zeros(self.num_classes, dtype=np.float32)
        onehot[target] = 1.0

        return sample, target, onehot


def get_mixup_fn(num_classes, args):
    if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
        # mixup is enabled
        fn = Mixup(mixup_alpha=args.mixup_alpha,
                  cutmix_alpha=args.cutmix_alpha,
                  label_smoothing=args.label_smooth,
                  num_classes=num_classes)
        def mixup_fn(images, labels, onehots):
            del onehots
            images, onehots = fn(images, labels)
            return images, labels, onehots
    elif args.label_smooth > 0:
        # apply label smoothing only
        def label_smooth_fn(images, labels, onehots):
            onehots = onehots * (1.0 - args.label_smooth) + args.label_smooth / num_classes
            return images, labels, onehots
        mixup_fn = label_smooth_fn
    else:
        mixup_fn = None

    return mixup_fn


def apply_mixup(batch, mixup_fn):
    if mixup_fn is not None:
        images, labels, onehots = batch  # ignore the 
        images, labels, onehots = mixup_fn(images, labels, onehots)
        return images, labels, onehots
    else:
        return batch

       
def prepare_batch_data(batch, batch_size=None):
  """Reformat a input batch from PyTorch Dataloader.
  
  Args:
    batch = (image, label)
      image: shape (host_batch_size, 3, height, width)
      label: shape (host_batch_size)
    batch_size = expected batch_size of this node, for eval's drop_last=False only
  """
  image, label, onehot = batch

  # pad the batch if smaller than batch_size
  if batch_size is not None and batch_size > image.shape[0]:
    image = torch.cat([image, torch.zeros((batch_size - image.shape[0],) + image.shape[1:], dtype=image.dtype)], axis=0)
    label = torch.cat([label, -torch.ones((batch_size - label.shape[0],), dtype=label.dtype)], axis=0)
    onehot = torch.cat([onehot, torch.zeros((batch_size - onehot.shape[0],) + onehot.shape[1:], dtype=onehot.dtype)], axis=0)

  # reshape (host_batch_size, 3, height, width) to
  # (local_devices, device_batch_size, height, width, 3)
  local_device_count = jax.local_device_count()
  image = image.permute(0, 2, 3, 1)
  image = image.reshape((local_device_count, -1) + image.shape[1:])
  label = label.reshape(local_device_count, -1)
  onehot = onehot.reshape((local_device_count, -1) + onehot.shape[1:])

  image = image.numpy()
  label = label.numpy()
  onehot = onehot.numpy()

  return_dict = {
    'image': image,
    'label': label,
    'onehot': onehot,
  }

  return return_dict


def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


from torchvision.datasets.folder import pil_loader
def loader(path: str):
    return pil_loader(path)


def build_transform(is_train, args):
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=IMAGE_SIZE,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.autoaug,
            scale=args.scale,
            ratio=args.ratio,
            interpolation=args.interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=MEAN_RGB,
            std=STDDEV_RGB,
        )
        return transform
    else:
      # eval transform: following DeiT
      t = []
      if IMAGE_SIZE <= 224:
          crop_pct = 224 / 256
      else:
          crop_pct = 1.0
      size = int(IMAGE_SIZE / crop_pct)
      t.append(
          transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
      )
      t.append(transforms.CenterCrop(IMAGE_SIZE))

      t.append(transforms.ToTensor())
      t.append(transforms.Normalize(MEAN_RGB, STDDEV_RGB))
      return transforms.Compose(t)


def create_split(
    dataset_cfg,
    batch_size,
    split,
):
  """Creates a split from the ImageNet dataset using Torchvision Datasets.

  Args:
    dataset_cfg: Configurations for the dataset.
    batch_size: Batch size for the dataloader.
    split: 'train' or 'val'.
  Returns:
    it: A PyTorch Dataloader.
    steps_per_epoch: Number of steps to loop through the DataLoader.
  """
  rank = jax.process_index()
  if split == 'train':
    transform = build_transform(is_train=True, args=dataset_cfg.aug)
    ds = ImageFolder(
      root=os.path.join(dataset_cfg.root, split),
      transform=transform,
      loader=loader,
    )
    logging.info(ds)
    sampler = DistributedSampler(
      ds,
      num_replicas=jax.process_count(),
      rank=rank,
      shuffle=True,
    )
    it = DataLoader(
      ds, batch_size=batch_size, drop_last=True,
      worker_init_fn=partial(worker_init_fn, rank=rank),
      sampler=sampler,
      num_workers=dataset_cfg.num_workers,
      prefetch_factor=dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None,
      pin_memory=dataset_cfg.pin_memory,
      persistent_workers=True if dataset_cfg.num_workers > 0 else False,
    )
    steps_per_epoch = len(it)
    mixup_fn = get_mixup_fn(ds.num_classes, dataset_cfg.aug)
  elif split == 'val':
    transform = build_transform(is_train=False, args=dataset_cfg.aug)
    ds = ImageFolder(
      root=os.path.join(dataset_cfg.root, split),
      transform=transform,
      loader=loader,
    )
    logging.info(ds)
    '''
    The val has 50000 images. We want to eval exactly 50000 images. When the
    batch is too big (>16), this number is not divisible by the batch size. We
    set drop_last=False and we will have a tailing batch smaller than the batch
    size, which requires modifying some eval code.
    '''
    sampler = DistributedSampler(
      ds,
      num_replicas=jax.process_count(),
      rank=rank,
      shuffle=False,  # don't shuffle for val
    )
    it = DataLoader(
      ds, batch_size=batch_size,
      drop_last=False,  # don't drop for val
      worker_init_fn=partial(worker_init_fn, rank=rank),
      sampler=sampler,
      num_workers=dataset_cfg.num_workers,
      prefetch_factor=dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None,
      pin_memory=dataset_cfg.pin_memory,
      persistent_workers=True if dataset_cfg.num_workers > 0 else False,
    )
    steps_per_epoch = len(it)
    mixup_fn = None
  else:
    raise NotImplementedError

  return it, steps_per_epoch, mixup_fn
