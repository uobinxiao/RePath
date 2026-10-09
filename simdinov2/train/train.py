# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from builtins import dict
import argparse
import hashlib
import json
import logging
import math
import os
from functools import partial
# import swanlab
import timm.optim
import torch
import sys
import timm
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')))
from simdinov2.utils.checkpoint import PeriodicCheckpointer
from simdinov2.data import SamplerType, make_data_loader, make_dataset
from simdinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator, WSIDataAugmentationDINO
from simdinov2.data.samplers import SemanticHierarchicalSampler
import simdinov2.distributed as dist
from simdinov2.fsdp import DCPCheckpointer as FSDPCheckpointer
from simdinov2.logging import MetricLogger
from simdinov2.utils.config import setup
from simdinov2.utils.utils import CosineScheduler
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from simdinov2.train.ssl_meta_arch_sim import SimSSLMetaArch
import pickle
torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def _sampler_type_from_config(cfg):
    sampler_cfg = getattr(cfg.train, "sampler", None)
    sampler_name = str(getattr(sampler_cfg, "type", "sharded_infinite")).strip().lower()
    sampler_types = {
        "infinite": SamplerType.INFINITE,
        "sharded_infinite": SamplerType.SHARDED_INFINITE,
        "sharded_infinite_new": SamplerType.SHARDED_INFINITE_NEW,
        "semantic_hierarchical": SamplerType.SEMANTIC_HIERARCHICAL,
    }
    if sampler_name not in sampler_types:
        raise ValueError(
            f"Unsupported train.sampler.type={sampler_name!r}; "
            f"expected one of {', '.join(sorted(sampler_types))}"
        )
    return sampler_types[sampler_name], sampler_cfg


def _sampling_contract_from_sampler(sampler, cfg):
    if not hasattr(sampler, "contract"):
        return None
    payload = {
        "schema_version": 1,
        **sampler.contract(),
        "batch_size_per_gpu": int(cfg.train.batch_size_per_gpu),
        "official_epoch_length": int(cfg.train.OFFICIAL_EPOCH_LENGTH),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["contract_id"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _sampling_contract(data_loader, cfg):
    return _sampling_contract_from_sampler(data_loader.sampler, cfg)


def _persist_sampling_contract(output_dir, contract):
    if contract is None:
        return
    path = os.path.join(output_dir, "sampling_contract.json")
    error = None
    if dist.is_main_process():
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if existing != contract:
                    raise RuntimeError(
                        f"Semantic sampling contract does not match the existing run: {path}"
                    )
            else:
                temporary = path + ".tmp"
                with open(temporary, "w", encoding="utf-8") as handle:
                    json.dump(contract, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                os.replace(temporary, path)
        except Exception as exception:
            error = str(exception)
    if dist.is_enabled():
        errors = [error]
        torch.distributed.broadcast_object_list(errors, src=0)
        error = errors[0]
    if error is not None:
        raise RuntimeError(error)
    with open(path, "r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    if persisted != contract:
        raise RuntimeError(f"Semantic sampling contract mismatch across ranks: {path}")


def _collective_run_checkpoint(checkpointer, resume):
    if not resume:
        return False
    local_marker = None
    if checkpointer.has_checkpoint():
        local_marker = os.path.basename(checkpointer.get_checkpoint_file())
    if not dist.is_enabled():
        return local_marker is not None
    markers = [None] * dist.get_global_size()
    torch.distributed.all_gather_object(markers, local_marker)
    if len(set(markers)) != 1:
        raise RuntimeError(
            "Distributed checkpoint markers are missing or inconsistent across ranks: "
            f"{markers}"
        )
    return local_marker is not None


def _verify_compiled_entry_files(semantic_index):
    error = None
    if dist.is_main_process():
        try:
            semantic_index.verify_entry_files()
        except Exception as exception:
            error = str(exception)
    if dist.is_enabled():
        errors = [error]
        torch.distributed.broadcast_object_list(errors, src=0)
        error = errors[0]
    if error is not None:
        raise RuntimeError(error)


def oom_observer(device, alloc, device_alloc, device_free):
    # snapshot right after an OOM happened
    print('saving allocated state during OOM')
    snapshot = torch.cuda.memory._snapshot()
    pickle.dump(snapshot, open('oom_snapshot.pickle', 'wb'))
    #python _memory_viz.py trace oom_snapshot.pickle

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--base-config", default="ssl_default_config", metavar="FILE", help="path to base config file")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--debug", action="store_true", help="debug flag")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, model):
    opt_lower = cfg.optim.opt.lower()
    params_groups = model.get_params_groups(True)
    opt = timm.optim.create_optimizer_v2(params_groups, opt=opt_lower, **cfg.optim.kwargs)
    return opt

def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        peak_iters=cfg.optim["peak_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.optim["min_lr"],
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )
    coeff = dict(
        base_value=cfg.dino.mcr.coeff,
        final_value=cfg.dino.mcr.coeff_end if cfg.dino.mcr.coeff_end>0 else cfg.dino.mcr.coeff,
        total_iters=(cfg.dino.mcr.expa_end_epoch if cfg.dino.mcr.expa_end_epoch>0 else cfg.optim["epochs"]) * OFFICIAL_EPOCH_LENGTH if cfg.dino.mcr.coeff_end>0 else 0,
        warmup_iters=0,
        start_warmup_value=0,
    )
    ibot_weight = dict(
        base_value=cfg.ibot.loss_weight,
        final_value=cfg.ibot.loss_weight_end if cfg.ibot.loss_weight_end > 0 else cfg.ibot.loss_weight,
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH if cfg.ibot.loss_weight_end > 0 else 0,
        warmup_iters=cfg.ibot.loss_weight_warmup_epochs* OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
        freeze_iters=cfg.ibot.loss_weight_freeze_epochs* OFFICIAL_EPOCH_LENGTH,
    )
    clip_grad = dict(
        base_value=cfg.optim.clip_grad,
        final_value=cfg.optim.clip_grad_end if cfg.optim.clip_grad_end>0 else cfg.optim.clip_grad,
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH if cfg.optim.clip_grad_end>0 else 0,
    )
    eps_schedule = dict(
        base_value=cfg.dino.mcr.eps,
        final_value=cfg.dino.mcr.eps_end if cfg.dino.mcr.eps_end>0 else cfg.dino.mcr.eps,
        total_iters=(cfg.dino.mcr.expa_end_epoch if cfg.dino.mcr.expa_end_epoch>0 else cfg.optim["epochs"]) * OFFICIAL_EPOCH_LENGTH if cfg.dino.mcr.eps_end>0 else 0,
        warmup_iters=0,
        start_warmup_value=0,
    )
    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(freeze_cut_iters=cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH,**lr)
    mcr_coeff_schedule = CosineScheduler(**coeff)
    mcr_eps_schedule = CosineScheduler(**eps_schedule)
    ibot_weight_schedule = CosineScheduler(**ibot_weight)
    clip_grad_schedule = CosineScheduler(**clip_grad)
    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
        mcr_coeff_schedule,
        mcr_eps_schedule,
        ibot_weight_schedule,
        clip_grad_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    new_state_dict = get_model_state_dict(model.teacher, options=StateDictOptions(full_state_dict=True, cpu_offload=False))
    if dist.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training


    #data_transform = DataAugmentationDINO(
    #    cfg.crops.global_crops_scale,
    #    cfg.crops.local_crops_scale,
    #    cfg.crops.local_crops_number,
    #    global_crops_size=cfg.crops.global_crops_size,
    #    local_crops_size=cfg.crops.local_crops_size,
    #)

    data_transform = WSIDataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        max_white_ratio=cfg.crops.get("max_white_ratio", 1.0),
        tissue_hsv_lower=cfg.crops.get("tissue_hsv_lower", (90, 8, 103)),
        tissue_hsv_upper=cfg.crops.get("tissue_hsv_upper", (180, 255, 255)),
        tissue_crop_max_attempts=cfg.crops.get("tissue_crop_max_attempts", 10),
    )

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
    )
    sampler_type, sampler_cfg = _sampler_type_from_config(cfg)
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    #for example, imagenet1k training set have 1281167 images. for 4nodes(32gpu) and 32 batch size per gpu, approximately 1250 iterations per epoch
    if OFFICIAL_EPOCH_LENGTH <= 0:
        if sampler_type == SamplerType.SEMANTIC_HIERARCHICAL:
            raise ValueError(
                "train.OFFICIAL_EPOCH_LENGTH must be explicitly positive for semantic hierarchical sampling"
            )
        OFFICIAL_EPOCH_LENGTH = len(dataset) // (cfg.train.batch_size_per_gpu * dist.get_global_size())
        print(f"OFFICIAL_EPOCH_LENGTH is not defined, set as {OFFICIAL_EPOCH_LENGTH} by dataset size and batch size")
        cfg.train.OFFICIAL_EPOCH_LENGTH = OFFICIAL_EPOCH_LENGTH
    preflight_sampling_contract = None
    if sampler_type == SamplerType.SEMANTIC_HIERARCHICAL:
        semantic_index = dataset.get_semantic_sampling_index()
        _verify_compiled_entry_files(semantic_index)
        preflight_sampler = SemanticHierarchicalSampler(
            semantic_index=semantic_index,
            magnification_weights=sampler_cfg.magnification_weights,
            bucket_weights=sampler_cfg.bucket_weights,
            seed=cfg.train.seed,
        )
        preflight_sampling_contract = _sampling_contract_from_sampler(preflight_sampler, cfg)
        _persist_sampling_contract(cfg.train.output_dir, preflight_sampling_contract)
    # setup optimizer
    optimizer = build_optimizer(cfg, model)
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
        mcr_coeff_schedule,
        mcr_eps_schedule,
        ibot_weight_schedule,
        clip_grad_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    start_iter = 0
    if cfg.MODEL.INIT_WEIGHTS != "":
        model.load_state_dict(torch.load(cfg.MODEL.INIT_WEIGHTS, map_location="cpu"), strict = False)
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    had_run_checkpoint = _collective_run_checkpoint(checkpointer, resume)
    checkpoint_state = {}
    try:
        checkpoint_state = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
        start_iter = checkpoint_state.get("iteration", -1) + 1
    except Exception as e:
        if sampler_type == SamplerType.SEMANTIC_HIERARCHICAL:
            raise RuntimeError("Failed to load a semantic-sampling training checkpoint") from e
        print("Failed to load checkpoint", e)
        traceback.print_exc()
        start_iter = 0
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    eval_period_iterations = cfg.evaluation.eval_period_epochs * OFFICIAL_EPOCH_LENGTH
    freeze_backbone = False
    freeze_backbone_iter  = cfg.student.freeze_backbone_epochs * OFFICIAL_EPOCH_LENGTH
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
        drop_masks=cfg.student.drop_masks,
    )

    # setup data loader
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=cfg.train.seed,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=start_iter * cfg.train.batch_size_per_gpu,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        sampler_config=sampler_cfg,
        drop_last=True,
        collate_fn=collate_fn,
    )
    sampling_contract = _sampling_contract(data_loader, cfg)
    if sampling_contract != preflight_sampling_contract:
        raise RuntimeError("Semantic sampling contract changed during data-loader setup")
    _persist_sampling_contract(cfg.train.output_dir, sampling_contract)
    checkpoint_contract = checkpoint_state.get("sampling_contract")
    checkpoint_has_progress = had_run_checkpoint or int(checkpoint_state.get("iteration", -1)) >= 0
    contract_required = checkpoint_contract is not None or (
        sampling_contract is not None and checkpoint_has_progress
    )
    if contract_required and checkpoint_contract != sampling_contract:
        raise RuntimeError(
            "Checkpoint semantic sampling contract does not match the current index, policy, or topology"
        )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"
    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        # apply schedules
        if "schedulefree" not in cfg.optim.opt.lower():
            lr = lr_schedule[iteration]
            wd = wd_schedule[iteration]
            last_layer_lr = last_layer_lr_schedule[iteration]
            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)
        else:
            lr = cfg.optim.lr
            wd = cfg.optim.weight_decay
            #last_layer_lr = cfg.optim.lr
            last_layer_lr = cfg.optim.lr if iteration > cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH else 0
            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)
            optimizer.train()
        if cfg.dino.use_mcr:
            mcr_coeff = mcr_coeff_schedule[iteration]
            model.dino_loss.coeff = mcr_coeff
            mcr_eps = mcr_eps_schedule[iteration]
            model.dino_loss.eps = mcr_eps
        if cfg.ibot.loss_weight >0:
            cfg.ibot.loss_weight = float(ibot_weight_schedule[iteration])
        if cfg.train.actions and (iteration % OFFICIAL_EPOCH_LENGTH==0):
            action_epochs, action_strs = zip(*cfg.train.actions)
            if iteration // OFFICIAL_EPOCH_LENGTH in action_epochs:
                exec(action_strs[action_epochs.index(iteration // OFFICIAL_EPOCH_LENGTH)])
        # compute losses

        optimizer.zero_grad(set_to_none=True)
        if freeze_backbone_iter > 0:
            # freeze_backbone 
            if iteration < freeze_backbone_iter:
                if not freeze_backbone:
                    for param in model.student.backbone.parameters():
                            param.requires_grad = False
                    freeze_backbone = True
                    logger.info(f"Freeze backbone at iter {iteration}")
            else:
                if freeze_backbone:
                    for param in model.student.backbone.parameters():
                        param.requires_grad = True
                    logger.info(f"Unfreeze backbone at iter {iteration}")
                    freeze_backbone = False
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)#, activate_ibot=activate_ibot)

        # clip gradients
        total_grad_norm = None
        if (clip_grad:=clip_grad_schedule[iteration]) > 0:
            if fp16_scaler is not None:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    total_grad_norm = v.clip_grad_norm_(clip_grad)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            else:
                for v in model.student.values():
                    total_grad_norm = v.clip_grad_norm_(clip_grad)
                optimizer.step()

        # perform teacher EMA update
        model.update_teacher(mom)

        # logging

        if math.isnan(sum(loss_dict.values())):
            print("loss_dict NaN detected: %s", loss_dict)
        if dist.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / dist.get_global_size() for k, v in loss_dict.items()}
        
        if math.isnan(sum(loss_dict_reduced.values())):
            print("loss_dict_reduced NaN detected: %s", loss_dict)
            if not cfg.ignore_nan:
                import debugpy
                # debugpy.listen(5678)
                # debugpy.wait_for_client()
                debugpy.breakpoint()
                raise AssertionError("NaN detected: %s", loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_dict_reduced |= dict(
            lr=lr,
            wd=wd,
            mom=mom,
            last_layer_lr=last_layer_lr,
            current_batch_size=current_batch_size,
            losses_reduced=losses_reduced,
            total_grad_norm=total_grad_norm
            )
        metric_logger.update(**loss_dict_reduced)
        if not cfg.debug and dist.is_main_process():
            # swanlab.log(loss_dict_reduced)
            pass

        # checkpointing and testing
        if "schedulefree" in cfg.optim.opt.lower():
            optimizer.eval()
        if eval_period_iterations > 0 and (iteration + 1) % eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        checkpoint_kwargs = {}
        if sampling_contract is not None:
            checkpoint_kwargs["sampling_contract"] = sampling_contract
        periodic_checkpointer.step(iteration, **checkpoint_kwargs)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

from torch.distributed.elastic.multiprocessing.errors import record
@record
def main(args):
    cfg = setup(args)
    cfg.debug = args.debug
    if dist.is_main_process():
        if args.debug:
            logger.info("========Debug Mode Enabled==========")
            torch.cuda.memory._record_memory_history()
            torch._C._cuda_attach_out_of_memory_observer(oom_observer)
        else:
            pass
            # swanlab.init(
            #     project="dinov2",
            #     logdir='./logs',
            #     mode="local",
            #     experiment_name=os.path.basename(cfg.train.output_dir),
            #     config=cfg)
    logger.info(f"Train with recipe:{cfg.train.recipe}")
    model = SimSSLMetaArch(cfg).to(torch.device("cuda"))
    logger.info(f"Init last layer weight:{model.student.backbone.blocks[-1][-1].mlp.fc2.state_dict()}")
    model.prepare_for_distributed_training()
    logger.info("Model:\n{}".format(model))
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    wdict= get_model_state_dict(model.student.backbone.module.blocks[-1].module[-1].mlp.fc2)
    logger.info(f"fsdp last layer state_dict:{wdict}")
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")
    if args.debug and dist.is_main_process():
        torch.cuda.memory._dump_snapshot("before_train.pickle")
        os.system("python _memory_viz.py before_train before_train.pickle")
    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
