'''
Adapted from https://github.com/facebookresearch/ToMe
'''

from .timesformer import apply_patch as timesformer
from .timesformer_prune import apply_patch as timesformer_prune
from .timesformer_merge import apply_patch as timesformer_merge
__all__ = ["timesformer", "timesformer_prune", "timesformer_merge"]