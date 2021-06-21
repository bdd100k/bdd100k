"""Transforms defined for the BDD100K dataset.

Code adapted from:
https://github.com/pytorch/vision/blob/master/references/detection/transforms.py

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

from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from torch import Tensor, nn
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T


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


class RandomHorizontalFlip(T.RandomHorizontalFlip):  # type: ignore
    """Randomly flip an input in the horizental direction."""

    def forward(  # pylint: disable=arguments-differ
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Executation function."""
        if torch.rand(1) < self.p:  # pylint: disable=no-member
            image = F.hflip(image)
            if target is not None:
                (
                    width,
                    _,
                ) = F._get_image_size(  # pylint: disable=protected-access
                    image
                )
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
        return image, target


class ToTensor(nn.Module):
    """Convert a np.ndarray into torch.Tensor."""

    def forward(  # pylint: disable=no-self-use
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Executation function."""
        image = F.to_tensor(image)
        return image, target


class RandomIoUCrop(nn.Module):
    """Randomly crop an input."""

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        """Init function."""
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Executation function."""
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {  # pylint: disable=no-else-raise
                2,
                3,
            }:
                raise ValueError(
                    "Should be 2/3 dimensional. Got {} dimensions.".format(
                        image.ndimension()
                    )
                )
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        orig_w, orig_h = F._get_image_size(  # pylint: disable=protected-access
            image
        )

        while True:
            # sample an option
            idx = int(
                torch.randint(  # pylint: disable=no-member
                    low=0, high=len(self.options), size=(1,)
                )
            )
            min_jaccard_overlap = self.options[idx]
            if (
                min_jaccard_overlap >= 1.0
            ):  # a value larger than 1 encodes the leave as-is option
                return image, target

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (
                    self.max_scale - self.min_scale
                ) * torch.rand(  # pylint: disable=no-member
                    2
                )
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (
                    self.min_aspect_ratio
                    <= aspect_ratio
                    <= self.max_aspect_ratio
                ):
                    continue

                # check for 0 area crops
                r = torch.rand(2)  # pylint: disable=no-member
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (
                    (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                )
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    boxes,
                    torch.tensor(  # pylint: disable=not-callable
                        [[left, top, right, bottom]],
                        dtype=boxes.dtype,
                        device=boxes.device,
                    ),
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                target["boxes"] = boxes
                target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)

                return image, target


class RandomZoomOut(nn.Module):
    """Randomly zoom out an input."""

    def __init__(
        self,
        fill: Optional[List[float]] = None,
        side_range: Tuple[float, float] = (1.0, 4.0),
        p: float = 0.5,
    ):
        """Init function."""
        super().__init__()
        if fill is None:
            fill = [0.0, 0.0, 0.0]
        self.fill = fill
        self.side_range = side_range
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(
                "Invalid canvas side range provided {}.".format(side_range)
            )
        self.p = p  # pylint: disable=invalid-name

    @torch.jit.unused
    def _get_fill_value(self, is_pil: bool) -> Union[Tuple[int, ...], int]:
        """Get fill value."""
        # We fake the type to make it work on JIT
        return tuple(int(x) for x in self.fill) if is_pil else 0

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Executation function."""
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {  # pylint: disable=no-else-raise
                2,
                3,
            }:
                raise ValueError(
                    "Should be 2/3 dimensional. Got {} dimensions.".format(
                        image.ndimension()
                    )
                )
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        if torch.rand(1) < self.p:  # pylint: disable=no-member
            return image, target

        orig_w, orig_h = F._get_image_size(  # pylint: disable=protected-access
            image
        )

        r = self.side_range[0] + torch.rand(1) * (  # pylint: disable=no-member
            self.side_range[1] - self.side_range[0]
        )
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)  # pylint: disable=no-member
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)

        if torch.jit.is_scripting():
            fill = 0
        else:
            fill = self._get_fill_value(
                F._is_pil_image(image)  # pylint: disable=protected-access
            )

        image = F.pad(image, [left, top, right, bottom], fill=fill)
        if isinstance(image, torch.Tensor):
            v = torch.tensor(  # pylint: disable=not-callable
                self.fill, device=image.device, dtype=image.dtype
            ).view(-1, 1, 1)
            image[..., :top, :] = image[..., :, :left] = image[
                ..., (top + orig_h) :, :
            ] = image[..., :, (left + orig_w) :] = v

        if target is not None:
            target["boxes"][:, 0::2] += left
            target["boxes"][:, 1::2] += top

        return image, target


class RandomPhotometricDistort(nn.Module):
    """Randoms apply photometric distort to an input."""

    def __init__(
        self,
        contrast: Tuple[float, float] = (0.5, 1.5),
        saturation: Tuple[float, float] = (0.5, 1.5),
        hue: Tuple[float, float] = (-0.05, 0.05),
        brightness: Tuple[float, float] = (0.875, 1.125),
        p: float = 0.5,
    ):
        """Init function."""
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p  # pylint: disable=invalid-name

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Executation function."""
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {  # pylint: disable=no-else-raise
                2,
                3,
            }:
                raise ValueError(
                    "Should be 2/3 dimensional. Got {} dimensions.".format(
                        image.ndimension()
                    )
                )
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)  # pylint: disable=no-member

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        if r[6] < self.p:
            channels = (
                F._get_image_num_channels(  # pylint: disable=protected-access
                    image
                )
            )
            permutation = torch.randperm(channels)  # pylint: disable=no-member

            is_pil = F._is_pil_image(image)  # pylint: disable=protected-access
            if is_pil:
                image = F.to_tensor(image)
            image = image[..., permutation, :, :]
            if is_pil:
                image = F.to_pil_image(image)

        return image, target
