import torch
from collections import OrderedDict
import copy
import torch.nn.functional as F

def interpolate_patch_weights(old_weight, new_size):
    in_channels = old_weight.shape[1]
    out_channels = old_weight.shape[0]

    # Reshape for interpolation: combine out/in channels into batch dimension
    reshaped_weight = old_weight.reshape(out_channels * in_channels, 1, old_weight.shape[2], old_weight.shape[3])

    # Interpolate to the new size (14x14)
    interpolated_weight = F.interpolate(reshaped_weight, size=new_size, mode='bilinear', align_corners=False)

    # Reshape back to the original format: [out_channels, in_channels, new_h, new_w]
    final_weight = interpolated_weight.reshape(out_channels, in_channels, new_size[0], new_size[1])

    print(old_weight.shape, final_weight.shape)

    return final_weight

def interpolate_vit_pos_embed(state_dict, model_pos_embed_key="pos_embed", target_num_tokens=None):
    """Interpolate pos_embed in state_dict to the current model's grid size."""
    pe_ckpt = state_dict[model_pos_embed_key]  # [1, 1+N, C]

    cls_tok = pe_ckpt[:, :1, :]
    grid_tok = pe_ckpt[:, 1:, :]               # [1, N, C]

    # Compute the side length of the checkpoint grid.
    N = grid_tok.shape[1]
    gs = int(N ** 0.5)
    assert gs * gs == N, f"pos_embed N={N} is not a perfect square; cannot reconstruct the grid"

    # Target token count obtained from the current model.
    if target_num_tokens is None:
        # For example, use the current model's parameter shape.
        # For a model not wrapped in FSDP: model.pos_embed.shape[1].
        raise ValueError("Please provide target_num_tokens (e.g. model.pos_embed.shape[1] for the current model)")

    tgt_grid = target_num_tokens - 1
    tgt_gs = int(tgt_grid ** 0.5)
    assert tgt_gs * tgt_gs == tgt_grid, "The target pos_embed token count is not 1 plus a perfect square"

    # Reshape to 4D, interpolate, then restore the token layout.
    grid_tok = grid_tok.transpose(1, 2).reshape(1, -1, gs, gs)          # [1, C, H, W]
    grid_tok = F.interpolate(grid_tok, size=(tgt_gs, tgt_gs), mode="bicubic", align_corners=False)
    grid_tok = grid_tok.reshape(1, -1, tgt_gs * tgt_gs).transpose(1, 2)  # [1, tgt_grid, C]

    pe_new = torch.cat([cls_tok, grid_tok], dim=1)                       # [1, 1+tgt_grid, C]
    state_dict[model_pos_embed_key] = pe_new
    return state_dict

def strip_prefix_if_present(state_dict, prefix="module."):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return OrderedDict((k[len(prefix):], v) if k.startswith(prefix) else (k, v)
                       for k, v in state_dict.items())


def rename_vit_to_framework_dinov2(vit_sd, target_prefix):
    out = OrderedDict()
    for k, v in vit_sd.items():
        new_k = f"{target_prefix}.backbone.{k}"
        out[new_k] = v
    return out

def rename_vit_to_framework_dino(vit_sd, target_prefix):
    out = OrderedDict()
    for k, v in vit_sd.items():
        if "head" in k:
            k = k.replace("head", "dino_head")
        new_k = f"{target_prefix}.{k}"
        out[new_k] = v
    return out


def torch_to_fvcore(in_path, out_path, key_in="state_dict", step_key=None):
    """
    Repackage a torch checkpoint to fvcore format:
    {
        "model": <state_dict>,
        "iteration": <int>,  # optional
        ...                  # you can add optimizer/scheduler too if you want
    }
    """

    ckpt = torch.load(in_path, map_location="cpu")

    # get raw state_dict (typical names: 'state_dict', 'model', or already flat)
    if isinstance(ckpt, dict) and key_in in ckpt:
        state = ckpt[key_in]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    else:
        # assume it's already a state_dict
        state = ckpt

    #for dinov2
    target_tokens = 257
    state = interpolate_vit_pos_embed(state, "pos_embed", target_num_tokens=target_tokens)
    state = strip_prefix_if_present(state, "module.")
    student_mapped = OrderedDict()
    student_mapped.update(rename_vit_to_framework_dinov2(state, "student"))
    teacher_mapped = OrderedDict()
    teacher_mapped.update(rename_vit_to_framework_dinov2(state, "teacher"))
    state = OrderedDict(**student_mapped, **teacher_mapped)

    #for dino patch size: 14, image size: 224
    #target_tokens = 257
    #state["student"] = strip_prefix_if_present(state["student"], "module.")
    #state["student"] = interpolate_vit_pos_embed(state["student"], "backbone.pos_embed", target_num_tokens=target_tokens)
    #state["student"]["backbone.patch_embed.proj.weight"] = interpolate_patch_weights(state["student"]["backbone.patch_embed.proj.weight"], (14, 14))
    #state["teacher"] = strip_prefix_if_present(state["teacher"], "module.")
    #state["teacher"] = interpolate_vit_pos_embed(state["teacher"], "backbone.pos_embed", target_num_tokens=target_tokens)
    #state["teacher"]["backbone.patch_embed.proj.weight"] = interpolate_patch_weights(state["teacher"]["backbone.patch_embed.proj.weight"], (14, 14))

    #for dino patch size: 16, image size: 224
    #target_tokens = 197
    #state["student"] = copy.deepcopy(state["teacher"])
    #state["student"] = strip_prefix_if_present(state["student"], "module.")
    #state["student"] = interpolate_vit_pos_embed(state["student"], "backbone.pos_embed", target_num_tokens=target_tokens)
    #state["student"]["backbone.patch_embed.proj.weight"] = interpolate_patch_weights(state["student"]["backbone.patch_embed.proj.weight"], (16, 16))
    #state["teacher"] = strip_prefix_if_present(state["teacher"], "module.")
    #state["teacher"] = interpolate_vit_pos_embed(state["teacher"], "backbone.pos_embed", target_num_tokens=target_tokens)
    #state["teacher"]["backbone.patch_embed.proj.weight"] = interpolate_patch_weights(state["teacher"]["backbone.patch_embed.proj.weight"], (16, 16))

    #student_mapped = OrderedDict()
    #student_mapped.update(rename_vit_to_framework_dinov2(state["student"], "student"))
    #teacher_mapped = OrderedDict()
    #teacher_mapped.update(rename_vit_to_framework_dinov2(state["teacher"], "teacher"))
    #state = OrderedDict(**student_mapped, **teacher_mapped)

    #fvcore_ckpt = {"model": state}
    ## carry over a training step if you have one
    #if step_key and step_key in ckpt and isinstance(ckpt[step_key], int):
    #    fvcore_ckpt["iteration"] = ckpt[step_key]

    torch.save(state, out_path)

    print(f"Saved fvcore-style checkpoint to: {out_path}")

if __name__ == "__main__":
    in_path = "dinov2_vitb14_pretrain.pth"
    out_path = "dinov2_vitb14_pretrain_renamed.pth"

    #in_path = "dinov2_vits14_pretrain.pth"
    #out_path = "dinov2_vits14_pretrain_renamed.pth"

    #in_path = "dino_vitbase16_pretrain_full_checkpoint.pth"
    #out_path = "dino_vitbase16_pretrained_renamed.pth"
    #in_path = "dino_vitbase16_pretrained_renamed.pth"
    #in_path = "dinov2_vitb14_pretrain.pth"
    torch_to_fvcore(in_path, out_path)
