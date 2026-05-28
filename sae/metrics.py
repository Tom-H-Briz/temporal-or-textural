"""
Sparsity and reconstruction metrics for SAE evaluation.
Local copy — originally from overcomplete.metrics.
"""

import torch


def l1(x):
    return x.abs().sum(-1)


def lp(x, p=2):
    return x.abs().pow(p).sum(-1).pow(1.0 / p)


def hoyer(x):
    l1_norm = x.abs().sum(-1)
    l2_norm = x.pow(2).sum(-1).sqrt()
    return l1_norm / (l2_norm + 1e-8)


def kappa_4(x):
    mu2 = x.pow(2).mean(-1)
    mu4 = x.pow(4).mean(-1)
    return mu4 / (mu2.pow(2) + 1e-8)


def dead_codes(codes):
    return (codes == 0).float()
