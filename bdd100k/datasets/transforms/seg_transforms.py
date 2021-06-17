"""Transforms defined for the BDD100K dataset.

Code adapted from:
https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py

Source License

BSD 3-Clause License

Copyright (c) 2017,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.s
############################################################################

Based on:
----------------------------------------------------------------------------
torchvision
Copyright (c) 2016 Facebook
Licensed under the BSD License [see LICENSE for details]
Written by the Pytorch Team
----------------------------------------------------------------------------
"""

import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T


def pad_if_smaller(img, size, fill=0):
    """Pad an image to the required size."""
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    """Compose transforms together."""

    def __init__(self, transforms):
        """Init function."""
        self.transforms = transforms

    def __call__(self, image, target):
        """Calling function."""
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    """Randomly resize an input give min_size and max_size."""

    def __init__(self, min_size, max_size=None):
        """Init function."""
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        """Calling function."""
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class RandomHorizontalFlip:
    """Randomly flip an input in the horizental direction."""

    def __init__(self, flip_prob):
        """Init function."""
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        """Calling function."""
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    """Randomly crop an input give the required size."""

    def __init__(self, size):
        """Init function."""
        self.size = size

    def __call__(self, image, target):
        """Calling function."""
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    """Centerly crop an input give the required size."""

    def __init__(self, size):
        """Init function."""
        self.size = size

    def __call__(self, image, target):
        """Calling function."""
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(nn.Module):
    """Convert a np.ndarray into torch.Tensor."""

    def forward(  # pylint: disable=no-self-use
        self, image: Tensor, target: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """Executation function."""
        image = F.to_tensor(image)
        target = torch.as_tensor(  # pylint: disable=no-member
            np.array(target), dtype=torch.int64  # pylint: disable=no-member
        )
        return image, target


class Normalize:
    """Normalize an image with mean and std."""

    def __init__(self, mean, std):
        """Init function."""
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        """Calling function."""
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
