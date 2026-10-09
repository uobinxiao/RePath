# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
logging.getLogger("pyvips").setLevel(logging.WARNING)
import os
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import h5py
from .extended import ExtendedVisionDataset
import glob
import openslide
import random
import json
import pyvips
from .wsi_utils import read_region
from ..semantic_index import SemanticSamplingIndex

logger = logging.getLogger("dinov2")

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class WSIPatch(ExtendedVisionDataset):

    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "WSIPatch.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._entries_meta = None
        self._entries_schema = None
        self._semantic_sampling_index = None
        self._class_ids = None
        self._class_names = None
        self.root = root

    @property
    def split(self) -> "WSIPatch.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str):
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path,  mmap_mode="r")

    def _load_extra_meta(self, extra_meta_path: str):
        extra_full_path = self._get_extra_full_path(extra_meta_path)
        with open(extra_full_path, "r", encoding="utf-8") as f:
            id2fname = json.load(f)

            return id2fname

    def _load_extra_schema(self, schema_path: str):
        schema_full_path = self._get_extra_full_path(schema_path)
        if not os.path.exists(schema_full_path):
            return {}
        with open(schema_full_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _split_name(self) -> str:
        return self._split.name if hasattr(self._split, "name") else str(self._split).upper()

    @property
    def _entries_path(self) -> str:
        #return f"entries-{self._split.upper()}_v2.npy"
        return f"entries-{self._split_name()}.npy"

    def _get_entries(self) -> List:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
            self._entries_meta = self._load_extra_meta(self._entries_path.replace(".npy", ".json"))
            self._entries_schema = self._load_extra_schema(self._entries_path.replace(".npy", ".schema.json"))
        assert self._entries is not None

        return self._entries

    def get_semantic_sampling_index(self) -> SemanticSamplingIndex:
        if self._semantic_sampling_index is None:
            semantic_index = SemanticSamplingIndex(self._extra_root, self._split_name())
            entries = self._get_entries()
            descriptor = semantic_index.manifest.get("entries", {})
            expected_shape = descriptor.get("shape")
            if semantic_index.entry_count != len(entries):
                raise ValueError(
                    "Semantic sampling entry count does not match "
                    f"{self._entries_path}: {semantic_index.entry_count} != {len(entries)}"
                )
            if list(entries.shape) != expected_shape or str(entries.dtype) != descriptor.get("dtype"):
                raise ValueError("WSIPatch entries do not match the compiled semantic manifest")
            self._semantic_sampling_index = semantic_index
        return self._semantic_sampling_index

    def _is_schema_v2_entry(self, entry) -> bool:
        if self._entries_schema and int(self._entries_schema.get("version", 1)) >= 2:
            return True
        return len(entry) >= 5

    #def _get_class_ids(self) -> np.ndarray:
    #    if self._class_ids is None:
    #        self._class_ids = self._load_extra(self._class_ids_path)
    #    assert self._class_ids is not None
    #    return self._class_ids

    #def _get_class_names(self) -> np.ndarray:
    #    if self._class_names is None:
    #        self._class_names = self._load_extra(self._class_names_path)
    #    assert self._class_names is not None
    #    return self._class_names

    def get_target(self, index: int):
        return 0
        #entries = self._get_entries()
        #class_index = entries[index]["class_index"]
        #return None if self.split == _Split.TEST else int(class_index)

    #def find_class_id(self, class_index: int) -> str:
    #    class_ids = self._get_class_ids()
    #    return str(class_ids[class_index])

    #def find_class_name(self, class_index: int) -> str:
    #    class_names = self._get_class_names()
    #    return str(class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        entry = self._get_entries()[index]
        svs_path = os.path.join(self.root, self._entries_meta[str(int(entry[2]))])
        patch_size = 256
        if self._is_schema_v2_entry(entry):
            level = 0
            read_size = int(entry[4])
        else:
            level = int(entry[3])
            read_size = patch_size
        #with openslide.OpenSlide(svs_path) as slide:
        #    image_data = slide.read_region((int(entry[0]), int(entry[1])), level, (patch_size, patch_size))

        cache_path = os.path.join("__REPATH_PRIVATE_SCRATCH_ROOT_006__", os.path.basename(svs_path))
        if os.path.exists(cache_path):
            try:
                image_data = read_region(cache_path, location = (int(entry[0]), int(entry[1])), level = level, size = (read_size, read_size))
            except:
                image_data = read_region(svs_path, location = (int(entry[0]), int(entry[1])), level = level, size = (read_size, read_size))
        else:
            image_data = read_region(svs_path, location = (int(entry[0]), int(entry[1])), level = level, size = (read_size, read_size))

        image_data = image_data.convert("RGB")
        if image_data.size != (patch_size, patch_size):
            image_data = image_data.resize((patch_size, patch_size))
        return image_data

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

    def _load_labels(self, labels_path: str) -> List[Tuple[str, str]]:
        labels_full_path = os.path.join(self.root, labels_path)
        labels = []

        try:
            with open(labels_full_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    class_id, class_name = row
                    labels.append((class_id, class_name))
        except OSError as e:
            raise RuntimeError(f'can not read labels file "{labels_full_path}"') from e

        return labels

    def _dump_entries(self) -> None:
        split = self.split
        if split == WSIPatch.Split.TEST:
            dataset = None
            sample_count = split.length
            max_class_id_length, max_class_name_length = 0, 0
        else:
            labels_path = "labels.txt"
            logger.info(f'loading labels from "{labels_path}"')
            labels = self._load_labels(labels_path)

            # NOTE: Using torchvision ImageFolder for consistency
            from torchvision.datasets import ImageFolder

            dataset_root = os.path.join(self.root, split.get_dirname())
            dataset = ImageFolder(dataset_root)
            sample_count = len(dataset)
            max_class_id_length, max_class_name_length = -1, -1
            for sample in dataset.samples:
                _, class_index = sample
                class_id, class_name = labels[class_index]
                max_class_id_length = max(len(class_id), max_class_id_length)
                max_class_name_length = max(len(class_name), max_class_name_length)

        dtype = np.dtype(
            [
                ("actual_index", "<u4"),
                ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_length}"),
                ("class_name", f"U{max_class_name_length}"),
            ]
        )
        entries_array = np.empty(sample_count, dtype=dtype)

        if split == WSIPatch.Split.TEST:
            old_percent = -1
            for index in range(sample_count):
                percent = 100 * (index + 1) // sample_count
                if percent > old_percent:
                    logger.info(f"creating entries: {percent}%")
                    old_percent = percent

                actual_index = index + 1
                class_index = np.uint32(-1)
                class_id, class_name = "", ""
                entries_array[index] = (actual_index, class_index, class_id, class_name)
        else:
            class_names = {class_id: class_name for class_id, class_name in labels}

            assert dataset
            old_percent = -1
            for index in range(sample_count):
                percent = 100 * (index + 1) // sample_count
                if percent > old_percent:
                    logger.info(f"creating entries: {percent}%")
                    old_percent = percent

                image_full_path, class_index = dataset.samples[index]
                image_relpath = os.path.relpath(image_full_path, self.root)
                class_id, actual_index = split.parse_image_relpath(image_relpath)
                class_name = class_names[class_id]
                entries_array[index] = (actual_index, class_index, class_id, class_name)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def _dump_class_ids_and_names(self) -> None:
        split = self.split
        if split == WSIPatch.Split.TEST:
            return

        entries_array = self._load_extra(self._entries_path)

        max_class_id_length, max_class_name_length, max_class_index = -1, -1, -1
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            max_class_index = max(int(class_index), max_class_index)
            max_class_id_length = max(len(str(class_id)), max_class_id_length)
            max_class_name_length = max(len(str(class_name)), max_class_name_length)

        class_count = max_class_index + 1
        class_ids_array = np.empty(class_count, dtype=f"U{max_class_id_length}")
        class_names_array = np.empty(class_count, dtype=f"U{max_class_name_length}")
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            class_ids_array[class_index] = class_id
            class_names_array[class_index] = class_name

        logger.info(f'saving class IDs to "{self._class_ids_path}"')
        self._save_extra(class_ids_array, self._class_ids_path)

        logger.info(f'saving class names to "{self._class_names_path}"')
        self._save_extra(class_names_array, self._class_names_path)

    def dump_extra(self) -> None:
        self._dump_entries()
        self._dump_class_ids_and_names()
