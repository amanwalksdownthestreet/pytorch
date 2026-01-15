# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Memory-saving distributed tensor utilities.

This module provides APIs for reducing per-device memory usage by physically
sharding tensor storage across devices in a process group. Unlike DTensor's
logical sharding (which affects the tensor's logical view), these utilities
partition the underlying storage itself.

Key APIs:
    scatter_storage: Scatter a single tensor's storage across ranks
    scatter_tensor_group: Scatter a group of tensors across ranks
    MemoryShardedDTensor: DTensor subclass with sharded storage
    TensorGroupStorage: Manage groups of sharded tensors
"""

from torch.distributed.tensor._memory_sharded import (
    MemoryShardedDTensor,
    scatter_storage,
    scatter_tensor_group,
    TensorGroupStorage,
)

__all__ = [
    "scatter_storage",
    "scatter_tensor_group",
    "MemoryShardedDTensor",
    "TensorGroupStorage",
]

# Set proper module names for public APIs
scatter_storage.__module__ = "torch.distributed.memory_saving"
scatter_tensor_group.__module__ = "torch.distributed.memory_saving"
MemoryShardedDTensor.__module__ = "torch.distributed.memory_saving"
TensorGroupStorage.__module__ = "torch.distributed.memory_saving"
