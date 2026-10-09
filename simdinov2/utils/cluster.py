# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ClusterType(Enum):
    AWS = "aws"
    FAIR = "fair"
    RSC = "rsc"


def _guess_cluster_type() -> ClusterType:
    uname = os.uname()
    if uname.sysname == "Linux":
        if uname.release.endswith("-aws"):
            # Linux kernel versions on AWS instances are of the form "5.4.0-1051-aws"
            return ClusterType.AWS
        elif uname.nodename.startswith("rsc"):
            # Linux kernel versions on RSC instances are standard ones but hostnames start with "rsc"
            return ClusterType.RSC

    return ClusterType.FAIR


def get_cluster_type(cluster_type: Optional[ClusterType] = None) -> Optional[ClusterType]:
    if cluster_type is None:
        return _guess_cluster_type()

    return cluster_type


def get_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[Path]:
    #cluster_type = get_cluster_type(cluster_type)
    #if cluster_type is None:
    #    return None

    #CHECKPOINT_DIRNAMES = {
    #    ClusterType.AWS: "checkpoints",
    #    ClusterType.FAIR: "checkpoint",
    #    ClusterType.RSC: "checkpoint/dino",
    #}
    #return Path(".") / CHECKPOINT_DIRNAMES[cluster_type]

    return Path("__REPATH_PRIVATE_PROJECT_ROOT_002__/dinov2_checkpoints/")


def get_user_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[Path]:
    checkpoint_path = get_checkpoint_path(cluster_type)
    if checkpoint_path is None:
        return None

    #username = os.environ.get("USER")
    #assert username is not None
    #return checkpoint_path / username

    return checkpoint_path


def get_slurm_executor_parameters(
    nodes: int,
    num_gpus_per_node: int,
    cpus_per_task: Optional[int] = None,
    mem_gb: Optional[int] = None,
    slurm_partition: Optional[str] = None,
    slurm_account: Optional[str] = None,
    cluster_type: Optional[ClusterType] = None,
    **kwargs,
) -> Dict[str, Any]:
    params = {
        "gpus_per_node": num_gpus_per_node,
        "tasks_per_node": num_gpus_per_node,  # one task per GPU
        "nodes": nodes,
    }
    optional_params = {
        "cpus_per_task": cpus_per_task,
        "mem_gb": mem_gb,
        "slurm_partition": slurm_partition,
        "slurm_account": slurm_account,
    }
    params.update({key: value for key, value in optional_params.items() if value not in (None, "")})
    print(params)
    params.update(kwargs)
    return params
