'''
Adapted from https://github.com/facebookresearch/ToMe
'''

import math
import os
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange


class _MergeWeightStats:
    """Collects and reports src/dst weight statistics across all merge calls."""
    def __init__(self):
        self.src_raw = []
        self.dst_raw = []
        self.src_w = []
        self.dst_w = []
        self.call_count = 0

    def record(self, src_raw, dst_raw, src_w, dst_w, alpha_src, alpha_dst):
        # src_raw/src_w: [B, N, 1, 1], dst_raw/dst_w: [B, 1, 1, 1]
        self.src_raw.append(src_raw.detach().float().mean().item())
        self.dst_raw.append(dst_raw.detach().float().mean().item())
        self.src_w.append(src_w.detach().float().mean().item())
        self.dst_w.append(dst_w.detach().float().mean().item())
        self.call_count += 1

    def summary(self):
        if not self.src_raw:
            return
        import numpy as np
        src_raw_np = np.array(self.src_raw)
        dst_raw_np = np.array(self.dst_raw)
        src_w_np = np.array(self.src_w)
        dst_w_np = np.array(self.dst_w)
        total_w = src_w_np + dst_w_np
        src_ratio = src_w_np / (total_w + 1e-8)
        # Debug summary (suppressed by default; enable by setting MERGE_WEIGHT_DEBUG=1)
        if os.environ.get('MERGE_WEIGHT_DEBUG', '0') == '1':
            print(f"[MergeWeight Summary] Total calls: {self.call_count}")
            print(f"  src_ratio (src_w/(src_w+dst_w)) - mean: {src_ratio.mean():.4f}, std: {src_ratio.std():.4f}")


_merge_stats = _MergeWeightStats()


def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching_TIM_TM(
    metric: torch.Tensor,
    score_obj: torch.Tensor = None,
    r: int = 1,
    class_token: bool = False,
    distill_token: bool = False,
    frame_average: bool = False,
    lambda_weight: float = 0.5,
    use_score_obj: bool = True,
    alpha_src: float = 0.0,
    alpha_dst: float = 0.0,
    cosine_threshold: float = -1,
) -> Tuple[Callable, Callable]:
    """
    [B, N, T, C] -> [B, N, T-r, C]
    """

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1


    t = metric.shape[-2]  # dimension for reduction
    r = min(r, (t - protected) // 1)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        # metric: [B, N, T, C]
        B, N, T, _ = metric.shape
        current_metric = metric.clone()
        if score_obj is None:
            current_score_obj = None
        else:
            current_score_obj = score_obj.clone()
        all_src_indices = []
        all_dst_indices = []
        all_max_sim_indices = []
        all_src_sos = []
        all_dst_sos = []
        all_cos_sim_token = []

        def min_max_norm(x):
            x_min = x.min(dim=1, keepdim=True).values
            x_max = x.max(dim=1, keepdim=True).values
            return (x - x_min) / (x_max - x_min + 1e-8)

        for i in range(r):
            _, _, T1, _ = current_metric.shape

            # 1. Cosine distance (mean across tokens)
            cos_sim = F.cosine_similarity(
                current_metric[:, :, :-1, :], current_metric[:, :, 1:, :], dim=-1
            )  # [B, N, T1-1]
            cos_dist = 1.0 - cos_sim.mean(dim=1)  # [B, T1-1]

            # 2. Attention scores (mean across tokens)
            if current_score_obj is not None:
                attn_scores = current_score_obj.squeeze(-1)  # [B, N, T1]
                attn_pair = (attn_scores[:, :, :-1] + attn_scores[:, :, 1:]) / 2.0  # [B, N, T1-1]
                attn_mean = attn_pair.mean(dim=1)  # [B, T1-1]
            else:
                attn_mean = torch.zeros(B, T1 - 1, device=metric.device)

            # 3. Min-max normalization
            normed_cos = min_max_norm(cos_dist)     # [B, T1-1]
            normed_attn = min_max_norm(attn_mean)   # [B, T1-1]

            # 4. Combined score, argmin to select merge pair
            combined = lambda_weight * normed_cos + (1 - lambda_weight) * normed_attn
            merge_idx = combined.argmin(dim=1)  # [B]

            # 5. Build indices (broadcast to [B, N, 1] for gather/scatter compatibility)
            max_similarity_indices = merge_idx.view(B, 1, 1).expand(-1, N, -1)  # [B, N, 1]

            # Calculate source and dst indices for compression
            src_indices = max_similarity_indices + 1
            dst_indices = torch.arange(T1 - 1).to(metric.device)[None, None, :].repeat(B, N, 1) #[B, N, T-1]
            dst_indices[dst_indices > max_similarity_indices] += 1

            src_so = None
            if current_score_obj is not None:
                C_s = current_score_obj.shape[-1]
                src_so = current_score_obj.gather(dim=2, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C_s))
                dst_so = current_score_obj.gather(dim=2, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C_s))
                all_dst_sos.append(dst_so.clone())  # store original dst_so BEFORE accumulation

                dst_so.scatter_add_(dim=2, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C_s), src=src_so)
                current_score_obj = dst_so

            # Save current merge indices
            all_src_indices.append(src_indices)
            all_dst_indices.append(dst_indices)
            all_max_sim_indices.append(max_similarity_indices)
            all_src_sos.append(src_so)

            # Merge current metric for next iteration
            C_m = current_metric.shape[-1]
            src_metric = current_metric.gather(dim=2, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C_m))
            dst_metric = current_metric.gather(dim=2, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C_m))

            # Precompute per-token cosine similarity for threshold check
            if cosine_threshold > -1:
                dst_at_merge = torch.gather(
                    dst_metric, 2,
                    max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C_m)
                )  # [B, N, 1, C_m]
                cos_sim_token = F.cosine_similarity(src_metric, dst_at_merge, dim=-1)  # [B, N, 1]
                all_cos_sim_token.append(cos_sim_token)
            else:
                all_cos_sim_token.append(None)
            dst_metric.scatter_add_(dim=2, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C_m), src=src_metric)
            current_metric = dst_metric

    def merge(x: torch.Tensor, mode="mean", apply_score_obj: bool = True,
              mlerp_target: bool = False) -> torch.Tensor:
        # x: [B, N, T, C]
        current_x = x.clone()
        C = x.shape[-1]
        for i in range(r):
            src_indices = all_src_indices[i]
            dst_indices = all_dst_indices[i]
            max_similarity_indices = all_max_sim_indices[i]
            idx = max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C)

            src = current_x.gather(dim=2, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C))
            dst = current_x.gather(dim=2, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C))

            if mlerp_target:
                dst_val = torch.gather(dst, 2, idx)
                max_norm = torch.max(src, dst_val)
                if all_cos_sim_token[i] is not None and cosine_threshold > -1:
                    accepted = (all_cos_sim_token[i] >= cosine_threshold).unsqueeze(-1)
                    target = torch.where(accepted, max_norm, torch.zeros_like(max_norm))
                else:
                    target = max_norm
                dst.scatter_(2, idx, target)
                current_x = dst
                continue

            if not apply_score_obj:
                dst = dst.scatter_reduce(dim=2, index=idx, src=src, reduce=mode)
                current_x = dst
                continue

            dst_val = torch.gather(dst, 2, idx)  # [B, N, 1, C] original dst at merge position
            src_orig = src.clone()  # preserve original src before weighted merge modifies it
            if use_score_obj and all_src_sos[i] is not None and (alpha_src != 0 or alpha_dst != 0):
                src_so = all_src_sos[i]  # [B, N, 1, 1]
                dst_so = all_dst_sos[i]  # [B, N, T-1, 1]
                C_s = dst_so.shape[-1]
                dst_so_at_merge = torch.gather(dst_so, 2, idx[:, :, 0:1, :C_s]).clamp(min=1e-6, max=1.0)
                src_w = alpha_src * src_so.clamp(min=1e-6, max=1.0)
                dst_w = alpha_dst * dst_so_at_merge
                _merge_stats.record(src_so, dst_so_at_merge, src_w, dst_w, alpha_src, alpha_dst)
                src = (src * src_w + dst_val * dst_w).to(dst.dtype)

            # Per-token cosine threshold + entropy-based selection
            if all_cos_sim_token[i] is not None:
                cos_sim_i = all_cos_sim_token[i]                        # [B, N, 1]
                below = (cos_sim_i < cosine_threshold).unsqueeze(-1)    # [B, N, 1, 1]

                if use_score_obj and all_src_sos[i] is not None:
                    src_so = all_src_sos[i]                              # [B, N, 1, 1]
                    dst_so = all_dst_sos[i]
                    idx_s = max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, dst_so.shape[-1])
                    dst_so_at_merge = torch.gather(dst_so, 2, idx_s)    # [B, N, 1, 1]
                    # lower score_obj = higher entropy = keep this token
                    keep_src = src_so < dst_so_at_merge                  # [B, N, 1, 1]
                    kept_val = torch.where(keep_src, src_orig, dst_val)  # [B, N, 1, C]
                else:
                    kept_val = dst_val                                   # [B, N, 1, C]

                src = torch.where(below, kept_val, src)  # [B, N, T-1, C]

            # For weighted/threshold cases: subtract dst_val at merge position so
            # scatter_reduce(include_self=True) cancels: dst_val + (merge_val - dst_val) = merge_val
            has_modified_src = (use_score_obj and all_src_sos[i] is not None and (alpha_src != 0 or alpha_dst != 0)) \
                               or all_cos_sim_token[i] is not None
            if has_modified_src:
                T_cur = src.shape[2]
                merge_mask = torch.zeros(1, 1, T_cur, 1, device=dst.device)
                merge_mask[:, :, 0:1, :] = 1.0
                src = src - dst_val * merge_mask
            dst = dst.scatter_reduce(dim=2, index=idx, src=src, reduce=mode)

            current_x = dst

        return dst
    return merge, None
