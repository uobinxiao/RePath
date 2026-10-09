import torch
from collections import OrderedDict
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

def rename_vit_to_framework_dinov2(vit_sd, target_prefix):
    out = OrderedDict()
    for k, v in vit_sd.items():
        new_k = f"{target_prefix}.{k}"
        out[new_k] = v
    return out

state_dict = torch.load("vitb16_reg4_SimDNIOv2_ep100.pth")
state = state_dict["teacher"]

target_tokens = 257
state = interpolate_vit_pos_embed(state, "backbone.pos_embed", target_num_tokens=target_tokens)
state["backbone.patch_embed.proj.weight"] = interpolate_patch_weights(state["backbone.patch_embed.proj.weight"], (14, 14))

student_mapped = OrderedDict()
student_mapped.update(rename_vit_to_framework_dinov2(state, "student"))
teacher_mapped = OrderedDict()
teacher_mapped.update(rename_vit_to_framework_dinov2(state, "teacher"))
state = OrderedDict(**student_mapped, **teacher_mapped)

torch.save(state, "simdinov2_vitb14.pth")
