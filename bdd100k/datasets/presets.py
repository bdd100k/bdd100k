"""Presetted transforms for the BDD100K dataset.

Code adapted from:
https://github.com/pytorch/vision/blob/master/references/detection/presets.py

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

from typing import Dict, Tuple

from torch import Tensor

from .transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomIoUCrop,
    RandomPhotometricDistort,
    RandomZoomOut,
    ToTensor,
)


class DetectionPresetTrain:
    """Preset transforms for BDD100K detection training."""

    def __init__(
        self,
        data_augmentation: str,
        hflip_prob: float = 0.5,
        mean: Tuple[float, float, float] = (123.0, 117.0, 104.0),
    ):
        """Init function."""
        if data_augmentation == "hflip":
            self.transforms = Compose(
                [
                    RandomHorizontalFlip(p=hflip_prob),
                    ToTensor(),
                ]
            )
        elif data_augmentation == "ssd":
            self.transforms = Compose(
                [
                    RandomPhotometricDistort(),
                    RandomZoomOut(fill=list(mean)),
                    RandomIoUCrop(),
                    RandomHorizontalFlip(p=hflip_prob),
                    ToTensor(),
                ]
            )
        elif data_augmentation == "ssdlite":
            self.transforms = Compose(
                [
                    RandomIoUCrop(),
                    RandomHorizontalFlip(p=hflip_prob),
                    ToTensor(),
                ]
            )
        else:
            raise ValueError(
                f'Unknown data augmentation policy "{data_augmentation}"'
            )

    def __call__(self, img: Tensor, target: Dict[str, Tensor]):
        """Calling function."""
        return self.transforms(img, target)


class DetectionPresetEval:
    """Preset transforms for BDD100K detection inference."""

    def __init__(self):
        """Init function."""
        self.transforms = ToTensor()

    def __call__(self, img: Tensor, target: Dict[str, Tensor]):
        """Calling function."""
        return self.transforms(img, target)
