'''
Adapted from https://github.com/facebookresearch/ToMe
'''

from .timesformer import apply_patch as timesformer
from .timesformer_prune import apply_patch as timesformer_prune
from .timesformer_testa import apply_patch as timesformer_testa
__all__ = ["timesformer", "timesformer_prune", "timesformer_testa"]