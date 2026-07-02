'''
Adapted from https://github.com/facebookresearch/ToMe
'''

import math
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import einsum


def do_nothing(x, mode=None):
    return x

#######################################################################################################
def bipartite_soft_matching_SIM_TM(
    x: torch.Tensor,
    size: torch.Tensor,
    source: torch.Tensor,
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    saliency_aware: bool = False,
    prune_high_entropy: bool = False,
    entropy_scores: torch.Tensor = None,
    entropy_prune_k: float = 1.0,
    protect_high_incoming: bool = False,
    incoming_attn_score: torch.Tensor = None,
    incoming_protect_k: float = 1.0,
    shared_merging: bool = False,
    token_norms: torch.Tensor = None,
) -> Tuple:

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1


    t = metric.shape[-2]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return x, size, source, token_norms

    def merging(data, unm_idx, src_idx, dst_idx, effective_r, mask, mode="mean"):
        dominant_tokens = data.masked_select(~mask.expand(-1,-1,-1,data.shape[3])).view(
            data.shape[0], data.shape[1], -1, data.shape[3])
        data_filtered = data.masked_select(mask.expand(-1,-1,-1,data.shape[3])).view(
                data.shape[0], data.shape[1], -1, data.shape[3])

        src, dst = data_filtered[..., ::2, :], data_filtered[..., 1::2, :]
        B, n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(B, n, t1 - effective_r, c))
        src = src.gather(dim=-2, index=src_idx.expand(B, n, effective_r, c))

        dst = dst.scatter_reduce(-2, dst_idx.expand(B, n, effective_r, c), src, reduce=mode)
        if distill_token:
            return torch.cat([dominant_tokens[:,:1], unm[:, :1], dst[:, :1], dominant_tokens[:,1:], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([dominant_tokens, unm, dst], dim=-2)

    def tome_merging(m_, x_, size_, source_, score_obj_=None, static_score=None,
                     entropy_scores_=None, incoming_attn_=None, norms_=None):
        assert class_token == True

        # shared_merging: average scores across frames to determine shared topology
        if shared_merging:
            if entropy_scores_ is not None:
                entropy_scores_used = entropy_scores_.mean(dim=1, keepdim=True)
            else:
                entropy_scores_used = None
            if incoming_attn_ is not None:
                incoming_attn_used = incoming_attn_.mean(dim=1, keepdim=True)
            else:
                incoming_attn_used = None
        else:
            entropy_scores_used = entropy_scores_
            incoming_attn_used = incoming_attn_

        # ===== Step 1: Entropy-based Pruning =====
        prune_count = 0
        if prune_high_entropy and entropy_scores_used is not None:
            ent_wocls = entropy_scores_used[:, :, 1:, 0]  # [B, W, K-1]
            ent_mean = ent_wocls.mean(dim=-1, keepdim=True)
            ent_std = ent_wocls.std(dim=-1, keepdim=True)
            threshold = ent_mean - entropy_prune_k * ent_std
            abnormal_mask = ent_wocls < threshold  # True = should be pruned

            max_prune = 2 * r
            abnormal_count = abnormal_mask.sum(dim=-1)  # [B, W]

            # Take min abnormal count across all (b, w), round down to even, clamp to max_prune
            actual_prune = abnormal_count.min().clamp(max=max_prune)
            prune_count = int((actual_prune // 2) * 2)

            if prune_count > 0:
                # Select wocls indices of prune_count tokens with lowest scores
                _, prune_indices_wocls = ent_wocls.topk(prune_count, dim=-1, largest=False)
                # Convert to global index (+1 because CLS is at position 0)
                prune_indices = prune_indices_wocls + 1  # [B, W, prune_count]

                # shared_merging: replicate shared indices to all frames
                if shared_merging and prune_indices.shape[1] == 1 and m_.shape[1] > 1:
                    prune_indices = prune_indices.repeat(1, m_.shape[1], 1)

                # Create keep mask: True = keep, False = prune
                keep_mask = torch.ones(m_.shape[0], m_.shape[1], m_.shape[2],
                                       dtype=torch.bool, device=m_.device)
                keep_mask.scatter_(2, prune_indices, False)
                keep_mask = keep_mask.unsqueeze(-1)  # [B, W, K, 1]

                K_after = m_.shape[2] - prune_count
                m_ = m_[keep_mask.expand_as(m_)].view(m_.shape[0], m_.shape[1], K_after, m_.shape[3])
                x_ = x_[keep_mask.expand_as(x_)].view(x_.shape[0], x_.shape[1], K_after, x_.shape[3])
                size_ = size_[keep_mask.expand_as(size_)].view(size_.shape[0], size_.shape[1], K_after, size_.shape[3])
                if source_ is not None:
                    src_keep = keep_mask.expand(-1, -1, -1, source_.shape[3])
                    source_ = source_[src_keep].view(source_.shape[0], source_.shape[1], K_after, source_.shape[3])
                if norms_ is not None:
                    norms_ = norms_[keep_mask.expand_as(norms_)].view(norms_.shape[0], norms_.shape[1], K_after, norms_.shape[3])
                if incoming_attn_used is not None:
                    inc_keep = keep_mask[:, :incoming_attn_used.shape[1]]
                    incoming_attn_used = incoming_attn_used[inc_keep.expand_as(incoming_attn_used)].view(
                        incoming_attn_used.shape[0], incoming_attn_used.shape[1], K_after, incoming_attn_used.shape[3])
                if static_score is not None:
                    static_score = static_score[keep_mask.expand_as(static_score)].view(
                        static_score.shape[0], static_score.shape[1], K_after, static_score.shape[3])

        effective_r = r - prune_count

        # ===== Step 2: Incoming Attention Protection =====
        if protect_high_incoming and incoming_attn_used is not None:
            inc_wocls = incoming_attn_used[:, :, 1:, 0]  # [B, W, K'-1]
            inc_mean = inc_wocls.mean(dim=-1, keepdim=True)
            inc_std = inc_wocls.std(dim=-1, keepdim=True)
            threshold_inc = inc_mean + incoming_protect_k * inc_std
            protect_mask_vals = inc_wocls > threshold_inc  # [B, W, K'-1], True = protect

            if shared_merging and protect_mask_vals.shape[1] == 1 and m_.shape[1] > 1:
                protect_mask_vals = protect_mask_vals.repeat(1, m_.shape[1], 1)

            # Unify protection count: take max across all (b, w)
            protect_counts = protect_mask_vals.sum(dim=-1)  # [B, W]
            max_protect = int(protect_counts.max().item())

            if max_protect > 0:
                # Select top max_protect tokens by incoming attention score
                topk_indices_wocls = inc_wocls.topk(max_protect, dim=-1, largest=True).indices + 1
                all_indices = torch.cat([
                    torch.zeros((topk_indices_wocls.shape[0], topk_indices_wocls.shape[1], 1),
                                dtype=topk_indices_wocls.dtype, device=topk_indices_wocls.device),
                    topk_indices_wocls
                ], dim=2)  # [B, W, 1+max_protect]
            else:
                all_indices = torch.zeros((m_.shape[0], m_.shape[1], 1), dtype=torch.long, device=m_.device)
        else:
            all_indices = torch.zeros((m_.shape[0], m_.shape[1], 1), dtype=torch.long, device=m_.device)

        # shared_merging: replicate shared indices to all frames
        if shared_merging and all_indices.shape[1] == 1 and m_.shape[1] > 1:
            all_indices = all_indices.repeat(1, m_.shape[1], 1)

        # Create mask: mark tokens to preserve (exclude from merging) as False
        mask = torch.ones_like(m_[:, :, :, 0], dtype=torch.bool, device=m_.device).scatter_(2, all_indices, False).unsqueeze(-1)

        # Ensure effective_r does not exceed available token count
        candidate_count = int(mask.sum(dim=2).min().item())
        effective_r = min(effective_r, candidate_count // 2)
        if effective_r <= 0:
            return x_, size_, source_, norms_

        ### Filter: filter out selected tokens, process remaining
        m_filtered = m_[mask.expand(-1,-1,-1,m_.shape[3])].view(
            m_.shape[0], m_.shape[1], -1, m_.shape[3])

        # shared_merging: normalize per-frame then average, compute shared scores on average
        if shared_merging:
            m_filtered_normed = m_filtered / m_filtered.norm(dim=-1, keepdim=True)
            m_filtered_shared = m_filtered_normed.mean(dim=1, keepdim=True)
        else:
            m_filtered_shared = m_filtered / m_filtered.norm(dim=-1, keepdim=True)

        a, b = m_filtered_shared[..., ::2, :], m_filtered_shared[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., effective_r:, :]
        src_idx = edge_idx[..., :effective_r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=-2)[0]

        merged_x_ = merging(x_, unm_idx, src_idx, dst_idx, effective_r, mask, mode="sum")
        merged_size_ = merging(size_, unm_idx, src_idx, dst_idx, effective_r, mask, mode="sum")
        if source_ is not None:
            merged_source_ = merging(source_, unm_idx, src_idx, dst_idx, effective_r, mask, mode="amax")
        else:
            merged_source_ = None

        if norms_ is not None:
            merged_max_norms_ = merging(norms_, unm_idx, src_idx, dst_idx, effective_r, mask, mode="amax")
        else:
            merged_max_norms_ = None

        return merged_x_, merged_size_, merged_source_, merged_max_norms_

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        B, T, N, C = metric.shape

        # Compute inter-frame similarity per token as temporal redundancy signal
        if T == 1:
            # Single frame case: use self-similarity
            frames_sim = torch.ones(B, T, N, 1, device=metric.device)
        else:
            frames_normed = F.normalize(metric, p=2, dim=-1)
            frames_sim = einsum('b w n c, b t n c -> b w t n', frames_normed, frames_normed)
            frames_sim = (frames_sim.sum(dim=-2) - 1).sum(dim=-2) / (T * (T - 1))  # B N
            frames_sim = frames_sim.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            frames_sim = frames_sim.repeat(1, T, 1, 1)  # [B, T, N, 1]

        merged_xs, merged_sizes, merged_sources, merged_max_norms = \
            tome_merging(metric, x, size, source, None,
                         static_score=frames_sim,
                         entropy_scores_=entropy_scores,
                         incoming_attn_=incoming_attn_score,
                         norms_=token_norms)

    return merged_xs, merged_sizes, merged_sources, merged_max_norms
