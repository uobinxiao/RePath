# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

#from torchvision.transforms import v2
from torchvision import transforms
from torchvision.transforms import functional as F

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
    RandomRotateList,
    RandomRotate90or270
)

from .hed_transform import HEDJitter
from .hsv_transform import HSVJitter
import cv2
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger("dinov2")


class TissueAwareRandomResizedCrop(transforms.RandomResizedCrop):
    """Random resized crop that prefers regions below a background ratio.

    Tissue pixels are selected with inclusive OpenCV HSV ranges. All remaining
    pixels are treated as background. Candidate crops whose background fraction
    is greater than or equal to ``max_white_ratio`` are rejected. If no
    candidate passes after ``max_attempts``, the candidate with the smallest
    background fraction is used. When ``max_white_ratio`` is ``None``, tissue
    filtering is disabled and the transform behaves like ``RandomResizedCrop``.
    """

    def __init__(
        self,
        *args,
        max_white_ratio=1.0,
        tissue_hsv_lower=(90, 8, 103),
        tissue_hsv_upper=(180, 255, 255),
        max_attempts=10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.max_white_ratio = None
        # When max_white_ratio is None, fall back to RandomResizedCrop.
        if max_white_ratio is not None:
            if not 0.0 < max_white_ratio <= 1.0:
                raise ValueError("max_white_ratio must be in (0, 1].")
            if max_attempts < 1:
                raise ValueError("max_attempts must be at least 1.")

            self.max_white_ratio = float(max_white_ratio)

            self.tissue_hsv_lower = self._validate_hsv_bound(
                tissue_hsv_lower,
                "tissue_hsv_lower",
            )
            self.tissue_hsv_upper = self._validate_hsv_bound(
                tissue_hsv_upper,
                "tissue_hsv_upper",
            )
            if np.any(self.tissue_hsv_lower > self.tissue_hsv_upper):
                raise ValueError("tissue_hsv_lower must not exceed tissue_hsv_upper.")
            self.max_attempts = int(max_attempts)

    @staticmethod
    def _validate_hsv_bound(bound, name):
        if len(bound) != 3:
            raise ValueError(f"{name} must contain exactly three values.")
        values = np.asarray(bound, dtype=np.int64)
        limits = np.asarray((180, 255, 255), dtype=np.int64)
        if np.any(values < 0) or np.any(values > limits):
            raise ValueError(f"{name} must use OpenCV HSV ranges H=[0, 180], S/V=[0, 255].")
        return values.astype(np.uint8)

    def make_background_integral(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("Tissue-aware cropping expects a PIL image.")

        rgb = np.asarray(image.convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        tissue_mask = cv2.inRange(
            hsv,
            self.tissue_hsv_lower,
            self.tissue_hsv_upper,
        ).astype(bool)
        background_mask = ~tissue_mask
        integral = np.zeros(
            (background_mask.shape[0] + 1, background_mask.shape[1] + 1),
            dtype=np.int64,
        )
        integral[1:, 1:] = background_mask.cumsum(axis=0).cumsum(axis=1)
        return integral

    @staticmethod
    def _background_ratio(background_integral, top, left, height, width):
        bottom = top + height
        right = left + width
        background_pixels = (
            background_integral[bottom, right]
            - background_integral[top, right]
            - background_integral[bottom, left]
            + background_integral[top, left]
        )
        return float(background_pixels) / float(height * width)

    def forward(self, image, background_integral=None):

        if self.max_white_ratio is None:
            return super().forward(image)

        if background_integral is None:
            background_integral = self.make_background_integral(image)

        best_params = None
        best_background_ratio = float("inf")
        for _ in range(self.max_attempts):
            params = self.get_params(image, self.scale, self.ratio)
            background_ratio = self._background_ratio(background_integral, *params)
            if background_ratio < best_background_ratio:
                best_params = params
                best_background_ratio = background_ratio
            if background_ratio < self.max_white_ratio:
                break

        top, left, height, width = best_params
        return F.resized_crop(
            image,
            top,
            left,
            height,
            width,
            self.size,
            self.interpolation,
            antialias=self.antialias,
        )

class _TissueAwareGeometricAugmentation:
    def __init__(self, crop, transforms_after_crop):
        self.crop = crop
        self.transforms_after_crop = transforms.Compose(transforms_after_crop)

    def __call__(self, image, background_integral):
        crop = self.crop(image, background_integral=background_integral)
        return self.transforms_after_crop(crop)


class WSIDataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        max_white_ratio=1.0,
        tissue_hsv_lower=(90, 8, 103),
        tissue_hsv_upper=(180, 255, 255),
        tissue_crop_max_attempts=10,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.max_white_ratio = max_white_ratio
        self.tissue_hsv_lower = tissue_hsv_lower
        self.tissue_hsv_upper = tissue_hsv_upper
        self.tissue_crop_max_attempts = tissue_crop_max_attempts

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"max_white_ratio: {max_white_ratio}")
        logger.info(f"tissue_hsv_lower: {tissue_hsv_lower}")
        logger.info(f"tissue_hsv_upper: {tissue_hsv_upper}")
        logger.info(f"tissue_crop_max_attempts: {tissue_crop_max_attempts}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = _TissueAwareGeometricAugmentation(
            TissueAwareRandomResizedCrop(
                global_crops_size,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
                max_white_ratio=max_white_ratio,
                tissue_hsv_lower=tissue_hsv_lower,
                tissue_hsv_upper=tissue_hsv_upper,
                max_attempts=tissue_crop_max_attempts,
            ),
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                RandomRotate90or270(p=0.5),
            ],
        )

        self.geometric_augmentation_local = _TissueAwareGeometricAugmentation(
            TissueAwareRandomResizedCrop(
                local_crops_size,
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
                max_white_ratio=max_white_ratio,
                tissue_hsv_lower=tissue_hsv_lower,
                tissue_hsv_upper=tissue_hsv_upper,
                max_attempts=tissue_crop_max_attempts,
            ),
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                RandomRotate90or270(p=0.5),
            ],
        )

        self.hed_jitter = HEDJitter(theta = 0.01, p = 0.1)

        #self.hsv_jitter = HSVJitter(
        #        hue_sigma_range = (-0.2, 0.2),
        #        saturation_sigma_range = (-0.2, 0.2),
        #        brightness_sigma_range = (-0.2, 0.2),
        #        p = 0.1,
        #        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        #global_transfo1_extra = GaussianBlurV2(p=1.0)
        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                #transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        #local_transfo_extra = GaussianBlurV2(p=0.5)
        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        #IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        #IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
                #v2.ToDtype(torch.float32, scale=True),
                #v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

        self.global_transfo1 = transforms.Compose([self.hed_jitter, color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([self.hed_jitter, color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([self.hed_jitter, color_jittering, local_transfo_extra, self.normalize])

        #self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        #self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        #self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}
        background_integral = None
        if self.max_white_ratio is not None:
            background_integral = self.geometric_augmentation_global.crop.make_background_integral(image)

        # global crops:
        im1_base = self.geometric_augmentation_global(image, background_integral)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image, background_integral)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        #output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image, background_integral))
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
