'''
Adapted from https://github.com/facebookresearch/ToMe
'''


from typing import Tuple
import torch.nn.functional as F
import torch
from model.surgformer_HTA import Attention_Spatial, Attention_Temporal, Block, VisionTransformer
from model.surgformer_HTA_KCA import (
    Block as KCA_Block,
    Attention_Spatial as KCA_Attention_Spatial,
    Attention_Temporal as KCA_Attention_Temporal,
)
from einops import rearrange
from ToMe.merge import bipartite_soft_matching, merge_source, merge_wavg
from ToMe.utils import parse_r, parse_merging_type
import math


class ToMeBlock(Block):
    """
    Modifications:
     - Apply token merging between the attention and mlp blocks
     - Compute and propagate token size and potentially the token sources.
    """
    def forward(self, x: torch.Tensor, B, T, K) -> torch.Tensor:
        """
        x: [bsz, 1+seq_len*n_frm, dim] for video
        """
        attn_size_s = self._tome_info["size_s"] if self._tome_info["prop_attn"] else None
        merging_type = self._tome_info["merging_type"].pop(0)
        self.attention_type = 'divided_space_time'
        if self.attention_type == 'divided_space_time':
            # Temporal_Self_Attention
            xt = x[:, 1:, :]  # [B, KxT, D]
            xt = rearrange(xt, "b (k t) c -> (b k) t c", t=T)
            xt_attn, metric_t = self.temporal_attn.forward(self.temporal_norm1(xt), B)
            res_temporal = self.drop_path(xt_attn)
            res_temporal = rearrange(res_temporal, '(b k) t c -> b (k t) c', b=B)
            xt = self.temporal_fc(res_temporal) + x[:, 1:, :]
            if 'frame' in merging_type:
                self._tome_info["class_token"] = False
                xt, rt = self.merge_tokens(xt, metric_t, B, K, 'frame')
                T = xt.size(1) // K

            # Spatial_Self_Attention
            init_cls_token = x[:, 0, :].unsqueeze(1)  # [B, 1, C]
            cls_token = init_cls_token.repeat(1, T, 1)  # [B, T, C]
            cls_token = rearrange(cls_token, 'b t c -> (b t) c', b=B, t=T).unsqueeze(1)  # [BxT, 1, C]
            xs = xt  # [B, KxT, C]
            xs = rearrange(xs, 'b (k t) c -> (b t) k c', t=T)
            xs = torch.cat((cls_token, xs), 1)  # [BxT, 1+K, D]
            x_attn, metric_s = self.attn.forward(self.norm1(xs), B, attn_size_s)  # compute metric for ToMe, apply proportional attention
            res_spatial = self.drop_path(x_attn)
            
            # Taking care of CLS token
            cls_token = res_spatial[:, 0, :]  # [BxT,  C]
            cls_token = rearrange(cls_token, '(b t) c -> b t c', b=B, t=T)  # [B, T, C]
            if hasattr(self, 'norm_cls'):
                cls_token = self.norm_cls(cls_token)
                target_token = cls_token[:, -1, :].unsqueeze(1)
                attn_cls = (target_token @ cls_token.transpose(-1, -2)).softmax(dim=-1)
                cls_token = attn_cls @ cls_token
            else:
                cls_token = torch.mean(cls_token, 1, True)
            res_spatial = res_spatial[:, 1:, :]  # [BxT, K, C]
            res_spatial = rearrange(res_spatial, '(b t) k c -> b (k t) c', b=B)
            res = res_spatial  # [B, LxT, D]
            x = xt  # [B, LxT, D], feature before spatial attn

            # Mlp
            x = rearrange((x + res), 'b (k t) c -> (b t) k c', b=B, k=K, t=T)  # [BxT, K, C]
            final_cls = init_cls_token + cls_token
            x = torch.cat((final_cls.repeat(x.size(0) // final_cls.size(0), 1, 1), x), 1)
            
            if 'patch' in merging_type:
                self._tome_info["class_token"] = True
                x, rs = self.merge_tokens(x, metric_s, B, K, 'patch')
                x = x[:, 1:, :]  # exclude [cls]
                if rs>0:    
                    self._tome_info["size_s"] = self._tome_info["size"]
            
            # reconstruct
            K = x.size(1)
            x = rearrange(x, '(b t) k c -> b (k t) c', b=B, k=K, t=T)
            x = torch.cat((final_cls, x), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x, T, K

    
    def merge_tokens(self, x, metric, B, L, merging_type):
        r = self._tome_info["r"].pop(0)
        if r > 0:
            if merging_type == 'patch':
                x = rearrange(x, "(b t) l d -> b t l d", b=B)
                metric = rearrange(metric, "(b t) l d -> b t l d", b=B)
            else:  # merging_type == 'frame'
                x = rearrange(x, "b (l t) d -> b l t d", l=L)
                metric = rearrange(metric, "(b l) t d -> b l t d", b=B)
                if self._tome_info["size"] is not None:
                    # by default, the size of self._tome_info["size"] is [b, t, l, d]
                    self._tome_info["size"] = self._tome_info["size"].permute(0, 2, 1, 3)
                    self._tome_info["size"] = self._tome_info["size"][:, 1:, ...]  # remove cls
            # Apply token merging here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
                merging_type,
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
            if merging_type == 'patch':
                x = rearrange(x, "b t l d -> (b t) l d", b=B)
            else:  # merging_type == 'frame'
                self._tome_info["size"] = self._tome_info["size"].permute(0, 2, 1, 3)
                size_cls = torch.ones(B, self._tome_info["size"].size(1), 1, 1).to(self._tome_info["size"])
                self._tome_info["size"] = torch.cat([size_cls, self._tome_info["size"]], dim=-2)  # add cls
                x = rearrange(x, "b l t d -> b (l t) d", l=L)
        return x, r


class ToMeAttention_Spatial(Attention_Spatial): #spatial attention
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """
    def forward(self, x, B, size=None):
        BT, K, C = x.shape
        T = BT // B
        qkv = self.qkv(x)
        # For Intra-Spatial: (BT, heads, K, C)
        # Atten: K*K, Values: K*C
        qkv = rearrange(
            qkv,
            "(b t) k (qkv num_heads c) -> qkv (b t) num_heads k c",
            t=T,
            qkv=3,
            num_heads=self.num_heads,
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply proportional attention
        if size is not None:
            size = rearrange(size, "b t l d -> (b t) l d" , b=B)
            if size.shape[0] == attn.shape[0]: 
                attn = attn + size.log()[:, None, None, :, 0]
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = rearrange(
            x,
            "(b t) num_heads k c -> (b t) k (num_heads c)",
            b=B,
        )
        x = self.proj(x)
        # Return k as well here
        return self.proj_drop(x), k.mean(1)

    

class ToMeAttention_Temporal(Attention_Temporal): #temporal attention
    """
    Modifications:
     - Return the mean of k over heads from attention
    """
    def forward(self, x, B):
        
        BK, T, C = x.shape #T:16; 15
        # t1 = T // 4
        # t2 = T // 2
        # x_4 = x[: , T-t1: , ]
        # x_8 = x[: , t2: , ]
        # Dynamically compute segment lengths and handle remainders
        t1 = max(1, T // 4)  # ensure at least 1 timestep
        t2 = max(1, T // 2)  # ensure at least 1 timestep
        
        # Adjust segment logic
        x_4 = x[:, -t1:, :]      # take last t1 timesteps
        x_8 = x[:, -t2:, :]      # take last t2 timesteps
        
        x_16 = x
        K = BK // B
        
        qkv_4 = self.qkv_4(x_4)

        qkv_4 = rearrange(
            qkv_4,
            "(b k) t (qkv num_heads c) -> qkv (b k) num_heads t c",
            k=K,
            qkv=3,
            num_heads=self.num_heads,
        )
        q_4, k_4, v_4 = (qkv_4[0], qkv_4[1], qkv_4[2])

        qkv_8 = self.qkv_8(x_8)

        qkv_8 = rearrange(
            qkv_8,
            "(b k) t (qkv num_heads c) -> qkv (b k) num_heads t c",
            k=K,
            qkv=3,
            num_heads=self.num_heads,
        )
        q_8, k_8, v_8 = (qkv_8[0], qkv_8[1], qkv_8[2])

        qkv_16 = self.qkv_16(x_16)

        qkv_16 = rearrange(
            qkv_16,
            "(b k) t (qkv num_heads c) -> qkv (b k) num_heads t c",
            k=K,
            qkv=3,
            num_heads=self.num_heads,
        )
        q_16, k_16, v_16 = (qkv_16[0], qkv_16[1], qkv_16[2])
        
        attn_4 = (q_4 @ k_4.transpose(-2, -1)) * self.scale
       
        attn_4 = attn_4.softmax(dim=-1)
        attn_4 = self.attn_drop(attn_4)
        x_4 = attn_4 @ v_4
        x_4 = rearrange(x_4, "(b k) num_heads t c -> (b k) t (num_heads c)", b=B)

        attn_8 = (q_8 @ k_8.transpose(-2, -1)) * self.scale
        
        attn_8 = attn_8.softmax(dim=-1)
        attn_8 = self.attn_drop(attn_8)
        x_8 = attn_8 @ v_8
        x_8 = rearrange(x_8, "(b k) num_heads t c -> (b k) t (num_heads c)", b=B)

        attn_16 = (q_16 @ k_16.transpose(-2, -1)) * self.scale
        # attn_16_ = attn_16.softmax(dim=-1).detach()
        
        attn_16 = attn_16.softmax(dim=-1)
        attn_16 = self.attn_drop(attn_16)
        x_16 = attn_16 @ v_16
        x_16 = rearrange(x_16, "(b k) num_heads t c -> (b k) t (num_heads c)", b=B)

        x_4 = self.proj_4(x_4)
        # # x_8[:, t1:, :] = 0.5 * x_8[:, t1:, :] + 0.5 * x_4
        # Dynamically adjust fusion position
        overlap_start_8 = max(0, x_8.shape[1] - t1)
        x_8[:, overlap_start_8:, :] = 0.5 * x_8[:, overlap_start_8:, :] + 0.5 * x_4

        x_8 = self.proj_8(x_8)
        # x_16[:, t2: , :] = 0.5 * x_16[:, t2: , :] + 0.5 * x_8
        overlap_start_16 = max(0, x_16.shape[1] - t2)
        x_16[:, overlap_start_16:, :] = 0.5 * x_16[:, overlap_start_16:, :] + 0.5 * x_8
        
        
        x_16 = self.proj_drop(self.proj_16(x_16))
        return x_16, k_16.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward_features(self, x):
            x = self.patch_embed(x) 
            # B, T, K, C
            B, T, K, C = x.size()
            W = int(math.sqrt(K))
            
            # Add Spatial Position Embedding
            x = rearrange(x, "b t k c -> (b t) k c")
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # BT, 1, C
            x = torch.cat((cls_tokens, x), dim=1)  # BT, HW+1, C  
            x = x + self.pos_embed  # BT, HW, C  
            x = self.pos_drop(x)
            
            # Add Temporal Position Embedding
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]  # remove cls_tokens
            x = rearrange(x, "(b t) k c -> (b k) t c", b=B)
            x = x + self.time_embed  # BK, T, C 
            x = self.time_drop(x)
            
            # Add CLS token
            x = rearrange(x, "(b k) t c -> b (k t) c", b=B)  # Spatial-Temporal tokens
            x = torch.cat((cls_tokens, x), dim=1)  # concat CLS token with spatiotemporal tokens
            
            # Attention blocks
            K = (x.size(1) - 1) // T
            for bidx, blk in enumerate(self.blocks):
                x, T, K = blk(x, B, T, K)

            x = self.norm(x)
            return x[:, 0]

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            r = self.r.copy() if isinstance(self.r, list) else self.r
            merging_type = self.merging_type.copy() if isinstance(self.merging_type, list) else self.merging_type
            self._tome_info["r"] = parse_r(len(self.blocks), r)
            self._tome_info["merging_type"] = parse_merging_type(len(self.blocks), merging_type)
            self._tome_info["size"] = None
            self._tome_info["size_s"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, merging_type: str = 'patch', num_patches: int = 196
):
    """
    Applies token merging to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For training and for evaluating MAE models off the shelf set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model.merging_type = merging_type
    model._tome_info = {
        "r": model.r,
        "size": None,
        "size_s": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None and 'patch' in merging_type,
        "distill_token": False,
        "num_patches": num_patches,
    }
    
    model.sample_idx = 0

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, (Block, KCA_Block)):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, (Attention_Spatial, KCA_Attention_Spatial)):
            module.__class__ = ToMeAttention_Spatial
        elif isinstance(module, (Attention_Temporal, KCA_Attention_Temporal)):
            module.__class__ = ToMeAttention_Temporal