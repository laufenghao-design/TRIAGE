import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
import utils
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from collections import OrderedDict
import math
from model.surgformer_HTA_KCA import VisionTransformer, _cfg
import ToMe


@register_model
def surgformer_HTA_KCA_ToMe(pretrained=False, pretrain_path=None, config=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()



    if pretrained:
        print("Load ckpt from %s" % pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = model.state_dict()
        if "model_state" in checkpoint.keys():
            checkpoint = checkpoint["model_state"]
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                # strip `model.` prefix
                name = k[6:] if k.startswith("model") else k
                new_state_dict[name] = v
            checkpoint = new_state_dict

            add_list = []
            for k in state_dict.keys():
                if "blocks" in k and "qkv_4" in k:
                    k_init = k.replace("qkv_4", "qkv")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "qkv_8" in k:
                    k_init = k.replace("qkv_8", "qkv")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "qkv_16" in k:
                    k_init = k.replace("qkv_16", "qkv")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "proj_4" in k:
                    k_init = k.replace("proj_4", "proj")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "proj_8" in k:
                    k_init = k.replace("proj_8", "proj")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "proj_16" in k:
                    k_init = k.replace("proj_16", "proj")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
            print(f"Adding keys from pretrained checkpoint:", ", ".join(add_list))
            remove_list = []
            for k in state_dict.keys():
                if (
                    ("head" in k or "patch_embed" in k)
                    and k in checkpoint
                    and k in state_dict
                    and checkpoint[k].shape != state_dict[k].shape
                ):
                    remove_list.append(k)
                    del checkpoint[k]
            print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))

            utils.load_state_dict(model, checkpoint)

        elif "model" in checkpoint.keys():
            checkpoint = checkpoint["model"]

            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                # strip `model.` prefix
                name = k[8:] if k.startswith("encoder") else k
                new_state_dict[name] = v
            checkpoint = new_state_dict

            add_list = []
            for k in state_dict.keys():
                if "blocks" in k and "qkv_4" in k and "temporal_attn" in k:
                    k_init = k.replace("qkv_4", "qkv")
                    k_init = k_init.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "qkv_8" in k and "temporal_attn" in k:
                    k_init = k.replace("qkv_8", "qkv")
                    k_init = k_init.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "qkv_16" in k and "temporal_attn" in k:
                    k_init = k.replace("qkv_16", "qkv")
                    k_init = k_init.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "proj_4" in k and "temporal_attn" in k:
                    k_init = k.replace("proj_4", "proj")
                    k_init = k_init.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "proj_8" in k and "temporal_attn" in k:
                    k_init = k.replace("proj_8", "proj")
                    k_init = k_init.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "proj_16" in k and "temporal_attn" in k:
                    k_init = k.replace("proj_16", "proj")
                    k_init = k_init.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "temporal_norm1" in k:
                    k_init = k.replace("temporal_norm1", "norm1")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)

            print("Adding keys from pretrained checkpoint:", ", ".join(add_list))

            remove_list = []
            for k in state_dict.keys():
                if (
                    ("head" in k or "patch_embed" in k)
                    and k in checkpoint
                    and k in state_dict
                    and checkpoint[k].shape != state_dict[k].shape
                ):
                    remove_list.append(k)
                    del checkpoint[k]

            print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))
            utils.load_state_dict(model, checkpoint)

        else:
            add_list = []
            for k in state_dict.keys():
                if "blocks" in k and "temporal_attn" in k:
                    k_init = k.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "temporal_norm1" in k:
                    k_init = k.replace("temporal_norm1", "norm1")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)

            print("Adding keys from pretrained checkpoint:", ", ".join(add_list))

            remove_list = []
            for k in state_dict.keys():
                if (
                    ("head" in k or "patch_embed" in k)
                    and k in checkpoint
                    and k in state_dict
                    and checkpoint[k].shape != state_dict[k].shape
                ):
                    remove_list.append(k)
                    del checkpoint[k]
            print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))
            utils.load_state_dict(model, checkpoint)



    if 'lambda_weight' in config:
        model.lambda_weight = config['lambda_weight']
    if 'cosine_threshold' in config:
        model.cosine_threshold = config['cosine_threshold']

    if 'entropy_prune_k' in config.keys():
        model.entropy_prune_k = config['entropy_prune_k']

    if 'incoming_protect_k' in config.keys():
        model.incoming_protect_k = config['incoming_protect_k']

    ToMe.patch.timesformer(model, trace_source=False, prop_attn=True,
                                        num_patches=model.patch_embed.num_patches)

    num_layers = len(model.blocks)
    r_temporal_val = config['r_temporal']
    r_spatial_val = config['r_spatial']
    model.r_temporal = [r_temporal_val] * 6 + [0] * (num_layers - 6)
    model.r_spatial = [0] * 6 + [r_spatial_val] * (num_layers - 6)

    print('config:', config)
    return model
