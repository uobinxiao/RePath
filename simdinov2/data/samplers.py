# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import itertools
import math
from bisect import bisect_right
from typing import Any, Mapping, Optional, Sequence
import warnings

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

import simdinov2.distributed as dist

from .semantic_index import SAMPLING_BUCKETS, SemanticSamplingIndex


class EpochSampler(Sampler):
    def __init__(
        self,
        *,
        size: int,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self._size = size
        self._sample_count = sample_count
        self._shuffle = shuffle
        self._seed = seed
        self._start = dist.get_global_rank() if start is None else start
        self._step = dist.get_global_size() if step is None else step
        self._epoch = 0

    def __iter__(self):
        count = (self._size + self._sample_count - 1) // self._sample_count
        tiled_indices = np.tile(np.arange(self._sample_count), count)
        if self._shuffle:
            seed = self._seed * self._epoch if self._seed != 0 else self._epoch
            rng = np.random.default_rng(seed)
            iterable = rng.choice(tiled_indices, self._size, replace=False)
        else:
            iterable = tiled_indices[: self._size]

        yield from itertools.islice(iterable, self._start, None, self._step)

    def __len__(self):
        return (self._size - self._start + self._step - 1) // self._step

    def set_epoch(self, epoch):
        self._epoch = epoch


def _get_numpy_dtype(size: int) -> Any:
    return np.int32 if size <= 2**31 else np.int64


def _get_torch_dtype(size: int) -> Any:
    return torch.int32 if size <= 2**31 else torch.int64


def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """Generate the indices of a random permutation."""
    dtype = _get_torch_dtype(size)
    # This is actually matching PyTorch's CPU implementation, see: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L900-L921
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()

        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value


class InfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = dist.get_global_rank() if start is None else start
        self._step = dist.get_global_size() if step is None else step
        self._advance = advance

    def __iter__(self):
        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator().manual_seed(self._seed)

        while True:
            iterable = _generate_randperm_indices(size=self._sample_count, generator=generator)
            yield from itertools.islice(iterable, self._start, None, self._step)


# The following function is somewhat equivalent to _new_shuffle_tensor_slice below,
# but avoids a full in-place random permutation generation.
def _shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}")

    dtype = _get_numpy_dtype(stop)
    result = np.empty(count, dtype=dtype)

    for i in range(count):
        j = torch.randint(0, i + 1, size=(1,), generator=generator).item() if i > 0 else 0

        result[i] = result[j]
        result[j] = tensor[start + i * step].item()

    return result


def _new_shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    dtype = torch.int64  # Needed for using randperm result as indices
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}")
    indices = torch.randperm(count, dtype=dtype, generator=generator)
    return tensor[start::step][indices].numpy()


def _make_seed(seed: int, start: int, iter_count: int) -> int:
    # NOTE: Tried a few variants (including iter_count << 32), this one worked best.
    return seed + start + (iter_count << 24)


class ShardedInfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
        use_new_shuffle_tensor_slice: bool = False,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = dist.get_global_rank() if start is None else start
        self._step = dist.get_global_size() if step is None else step
        self._advance = advance
        self._iter_count = 0
        self._shuffle_tensor_slice_fn = (
            _new_shuffle_tensor_slice if use_new_shuffle_tensor_slice else _shuffle_tensor_slice
        )

    def __iter__(self):
        iter_count = self._advance // self._sample_count
        if iter_count > 0:
            self._advance -= iter_count * self._sample_count
            self._iter_count += iter_count

        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to be keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator()

        # Always shuffle everything first
        generator.manual_seed(self._seed)
        dtype = _get_torch_dtype(self._sample_count)
        perm = torch.randperm(self._sample_count, dtype=dtype, generator=generator)

        while True:
            # Re-seed on each iteration to allow skipping whole permutations
            seed = _make_seed(self._seed, self._start, self._iter_count)
            generator.manual_seed(seed)

            iterable = self._shuffle_tensor_slice_fn(
                tensor=perm, start=self._start, step=self._step, generator=generator
            )
            yield from iterable
            self._iter_count += 1


_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _counter_random(seed: int, draw_id: int, stream: int) -> int:
    value = int(seed) & _UINT64_MASK
    value ^= (int(draw_id) * 0xD2B74407B1CE6E93) & _UINT64_MASK
    value ^= (int(stream) * 0xCA5A826395121157) & _UINT64_MASK
    return _splitmix64(value)


def _uniform_index(random_value: int, size: int) -> int:
    if size <= 0:
        raise ValueError("Cannot sample from an empty collection")
    return (int(random_value) * int(size)) >> 64


def _normalize_weights(
    values: Mapping[Any, Any],
    *,
    key_parser,
    label: str,
) -> tuple[tuple[Any, ...], tuple[float, ...], dict[Any, float]]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{label} must be a non-empty mapping")
    parsed: dict[Any, float] = {}
    for raw_key, raw_weight in values.items():
        key = key_parser(raw_key)
        if key in parsed:
            raise ValueError(f"{label} contains duplicate key {key}")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{label}[{key}] must be finite and non-negative")
        if weight > 0:
            parsed[key] = weight
    if not parsed:
        raise ValueError(f"{label} has no positive weights")
    total = sum(parsed.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{label} must sum to 1.0, got {total}")
    # Mapping insertion order is not part of the sampling policy. Canonicalize
    # keys so equivalent YAML/CLI mappings produce the same index stream.
    keys = tuple(sorted(parsed))
    cumulative = []
    running = 0.0
    for key in keys:
        running += parsed[key]
        cumulative.append(running)
    cumulative[-1] = 1.0
    return keys, tuple(cumulative), parsed


def _weighted_choice(keys: Sequence[Any], cumulative: Sequence[float], random_value: int) -> Any:
    unit = int(random_value) / float(1 << 64)
    index = min(bisect_right(cumulative, unit), len(keys) - 1)
    return keys[index]


class SemanticHierarchicalSampler(Sampler):
    """Deterministic online sampler over an approved compiled semantic index.

    Each draw selects magnification, semantic bucket, coarse cluster, fine
    cluster, slide, then tile. Draws use replacement. The random stream is a
    pure function of ``(seed, global_draw_id)`` so resuming does not replay the
    prefix of the stream.
    """

    def __init__(
        self,
        *,
        semantic_index: SemanticSamplingIndex,
        magnification_weights: Mapping[Any, Any],
        bucket_weights: Mapping[Any, Any],
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
    ) -> None:
        self.semantic_index = semantic_index
        self._seed = int(seed)
        self._start = dist.get_global_rank() if start is None else int(start)
        self._step = dist.get_global_size() if step is None else int(step)
        self._advance = int(advance)
        if self._step <= 0 or not 0 <= self._start < self._step:
            raise ValueError(f"Invalid distributed sampler start/step: {self._start}/{self._step}")
        if self._advance < 0:
            raise ValueError("sampler advance must be non-negative")

        self._magnifications, self._magnification_cdf, self.magnification_weights = _normalize_weights(
            magnification_weights,
            key_parser=lambda value: int(value),
            label="magnification_weights",
        )
        self._buckets, self._bucket_cdf, self.bucket_weights = _normalize_weights(
            bucket_weights,
            key_parser=lambda value: str(value),
            label="bucket_weights",
        )
        unknown_buckets = sorted(set(self._buckets) - set(SAMPLING_BUCKETS))
        if unknown_buckets:
            raise ValueError(f"Unsupported semantic sampling bucket(s): {', '.join(unknown_buckets)}")
        missing_buckets = sorted(set(self._buckets) - set(self.semantic_index.bucket_bits))
        if missing_buckets:
            raise ValueError(
                "Semantic index is missing configured semantic bucket(s): "
                + ", ".join(missing_buckets)
                + ". Recompile the offline sampling index before enabling them."
            )

        self._cells: dict[tuple[int, str], tuple[np.ndarray, ...]] = {}
        self._build_cells()

    def _build_cells(self) -> None:
        records = np.asarray(self.semantic_index.fine_records, dtype=np.int64)
        masks = np.asarray(self.semantic_index.bucket_masks, dtype=np.uint8)
        available_magnifications = set(int(value) for value in records[:, 0])
        missing_magnifications = sorted(set(self._magnifications) - available_magnifications)
        if missing_magnifications:
            raise ValueError(f"Semantic index is missing magnification(s): {missing_magnifications}")

        empty_cells = []
        for magnification in self._magnifications:
            mag_mask = records[:, 0] == int(magnification)
            for bucket in self._buckets:
                bit = np.uint8(1 << self.semantic_index.bucket_bits[bucket])
                fine_indices = np.flatnonzero(mag_mask & ((masks & bit) != 0))
                if len(fine_indices) == 0:
                    empty_cells.append(f"{magnification}x/{bucket}")
                    continue
                coarse_groups = []
                for coarse_id in np.unique(records[fine_indices, 1]):
                    group = fine_indices[records[fine_indices, 1] == coarse_id]
                    coarse_groups.append(group.astype(np.int64, copy=False))
                self._cells[(int(magnification), bucket)] = tuple(coarse_groups)
        if empty_cells:
            raise ValueError(
                "Positive-weight semantic bucket is empty for enabled magnification(s): "
                + ", ".join(empty_cells)
            )

    def sample_at_global_draw(self, draw_id: int) -> int:
        if draw_id < 0:
            raise ValueError("global draw ID must be non-negative")
        magnification = _weighted_choice(
            self._magnifications,
            self._magnification_cdf,
            _counter_random(self._seed, draw_id, 0),
        )
        bucket = _weighted_choice(
            self._buckets,
            self._bucket_cdf,
            _counter_random(self._seed, draw_id, 1),
        )
        coarse_groups = self._cells[(int(magnification), str(bucket))]
        coarse_group = coarse_groups[
            _uniform_index(_counter_random(self._seed, draw_id, 2), len(coarse_groups))
        ]
        fine_record_index = int(
            coarse_group[_uniform_index(_counter_random(self._seed, draw_id, 3), len(coarse_group))]
        )

        slide_start = int(self.semantic_index.fine_slide_offsets[fine_record_index])
        slide_end = int(self.semantic_index.fine_slide_offsets[fine_record_index + 1])
        slide_group = slide_start + _uniform_index(
            _counter_random(self._seed, draw_id, 4), slide_end - slide_start
        )
        entry_start = int(self.semantic_index.slide_entry_offsets[slide_group])
        entry_end = int(self.semantic_index.slide_entry_offsets[slide_group + 1])
        return entry_start + _uniform_index(
            _counter_random(self._seed, draw_id, 5), entry_end - entry_start
        )

    def __iter__(self):
        local_position = self._advance
        while True:
            global_draw_id = local_position * self._step + self._start
            yield self.sample_at_global_draw(global_draw_id)
            local_position += 1

    def contract(self) -> dict[str, Any]:
        return {
            **self.semantic_index.contract(),
            "sampler": "semantic_hierarchical",
            "sampler_schema_version": 1,
            "seed": self._seed,
            "world_size": self._step,
            "magnification_weights": {str(key): self.magnification_weights[key] for key in self._magnifications},
            "bucket_weights": {key: self.bucket_weights[key] for key in self._buckets},
        }
