"""
Local copy of overcomplete SAE library (KempnerInstitute/overcomplete).
Only the SAE-relevant modules are included.
"""

__version__ = '0.3.0'

from .base import SAE
from .batchtopk_sae import BatchTopKSAE
from .topk_sae import TopKSAE
from .jump_sae import JumpSAE
from .losses import top_k_auxiliary_loss, reanimation_regularizer, mse_l1
from .dictionary import DictionaryLayer
from .factory import EncoderFactory
from .modules import MLPEncoder, ResNetEncoder, AttentionEncoder
from .metrics import l1, lp, hoyer, kappa_4, dead_codes
