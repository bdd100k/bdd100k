"""Presetted transforms for the BDD100K dataset.

Code adapted from:
https://github.com/pytorch/vision/blob/master/references/segmentation/presets.py

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

from .seg_transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResize,
    ToTensor,
)


class SegmentationPresetTrain:
    """Preset transforms for BDD100K segmentation training."""

    def __init__(
        self,
        base_size,
        crop_size,
        hflip_prob=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        """Init function."""
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                RandomCrop(crop_size),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = Compose(trans)

    def __call__(self, img, target):
        """Init function."""
        return self.transforms(img, target)


class SegmentationPresetEval:
    """Preset transforms for BDD100K segmentation inference."""

    def __init__(
        self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    ):
        """Init function."""
        self.transforms = Compose(
            [
                RandomResize(base_size, base_size),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img, target):
        """Init function."""
        return self.transforms(img, target)
