# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from .image_net import ImageNet
from .image_net21k import ImageNet21k
from .wsi_utils import use_local_vips_linux, tune_vips_cache

VIPS_PREFIX = "__REPATH_PRIVATE_HOME_001__/libvips/bin"
OPENSLIDE_PREFIX = "__REPATH_PRIVATE_HOME_001__/openslide/bin"
DICOM_PREFIX = "__REPATH_PRIVATE_HOME_001__/libdicom/bin"
use_local_vips_linux(VIPS_PREFIX, OPENSLIDE_PREFIX, DICOM_PREFIX)

from .wsi_patch import WSIPatch
