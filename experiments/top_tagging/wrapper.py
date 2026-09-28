"""GATr wrapper for binary jet classification (top tagging).

Architecture:
  embed -> GATr -> extract grade-0 from MV output -> mean pool per jet -> logit
"""

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter


def build_attention_mask(batch: Tensor, dtype: torch.dtype) -> Tensor:
    """Block-diagonal additive float attention mask.

    Tokens in the same jet can attend to each other.
    Tokens in different jets cannot attend to each other.

    Parameters
    ----------
    batch : Tensor shape (n_total,) -- jet index per token
    dtype : float dtype for the mask

    Returns
    -------
    mask : Tensor shape (n_total, n_total)
        0.0 where tokens can attend; float("-inf") where blocked.
    """
    same_jet = batch.unsqueeze(0) == batch.unsqueeze(1)  # (n_total, n_total)
    mask = torch.full(
        (len(batch), len(batch)), float("-inf"),
        dtype=dtype, device=batch.device,
    )
    mask[same_jet] = 0.0
    return mask


def run_grouped(net: nn.Module, mv: Tensor, batch: Tensor, patch_mv: Tensor = None) -> Tensor:
    """Run GATr with attention restricted to each jet; mean-pool grade 0.

    Passes the jet index `batch` as the attention mask: sdp_attention then pads
    the jets into blocks for attention only (cost sum_jets L_max^2 instead of
    n_total^2 for a dense block-diagonal mask), while every per-token layer
    runs on the real tokens. Logits equal the dense-mask version
    (tests/test_wrapper_padded_attention.py).

    Parameters
    ----------
    net      : GATr
    mv       : Tensor (n_total, C, 8), tokens sorted by jet
    batch    : Tensor (n_total,) jet index per token
    patch_mv : optional Tensor (n_total, K, 8)

    Returns
    -------
    logits : Tensor (batchsize, 1)
    """
    extra = {} if patch_mv is None else {"patch_mv": patch_mv.unsqueeze(0)}
    mv_out, _ = net(mv.unsqueeze(0), scalars=None, attention_mask=batch, **extra)
    scalars_out = mv_out[0, :, :, 0]                          # (n_total, 1)
    batchsize = int(batch.max().item()) + 1
    return scatter(scalars_out, batch, dim=0, dim_size=batchsize, reduce="mean")


class TopTaggingWrapper(nn.Module):
    """GATr + mean-pool head for binary jet classification.

    Parameters
    ----------
    net : GATr configured with in_mv_channels=1, out_mv_channels=1,
          in_s_channels=None, out_s_channels=None
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, embedding: dict) -> Tensor:
        """Classify jets.

        Parameters
        ----------
        embedding : dict from embed_tagging_data
            "mv"      : Tensor (n_total, 1, 8)
            "batch"   : Tensor (n_total,) jet index per token
            "is_beam" : Tensor (n_total,) bool  (not used for masking here)

        Returns
        -------
        logits : Tensor (batchsize, 1)
        """
        # grade 0 (Spin(3)-invariant) is the classification signal
        return run_grouped(self.net, embedding["mv"], embedding["batch"])


class TopTaggingWrapperBivectorPatch(nn.Module):
    """GATr + mean-pool head for bivector patch-injected binary jet classification.

    Parameters
    ----------
    net : GATr configured with in_mv_channels=1, out_mv_channels=1,
          in_s_channels=None, out_s_channels=None, patch_mv_channels=K
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, embedding: dict) -> Tensor:
        """Classify jets.

        Parameters
        ----------
        embedding : dict from embed_tagging_data_bivector_patch
            "mv"       : Tensor (n_total, 1, 8)
            "patch_mv" : Tensor (n_total, K, 8)
            "batch"    : Tensor (n_total,) jet index per token
            "is_beam"  : Tensor (n_total,) bool (not used for masking here)

        Returns
        -------
        logits : Tensor (batchsize, 1)
        """
        return run_grouped(self.net, embedding["mv"], embedding["batch"],
                          patch_mv=embedding["patch_mv"])
