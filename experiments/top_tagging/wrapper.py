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
        mv = embedding["mv"].unsqueeze(0)   # (1, n_total, 1, 8)
        batch = embedding["batch"]          # (n_total,)

        mask = build_attention_mask(batch, mv.dtype)   # (n_total, n_total)

        mv_out, _ = self.net(mv, scalars=None, attention_mask=mask)
        # mv_out: (1, n_total, out_mv_channels=1, 8)

        # Extract grade-0 (Spin(3)-invariant) as classification signal
        scalars_out = mv_out[0, :, :, 0]  # (n_total, 1)

        batchsize = int(batch.max().item()) + 1
        logits = scatter(scalars_out, batch, dim=0, dim_size=batchsize, reduce="mean")
        return logits  # (batchsize, 1)


class TopTaggingWrapperR7Patch(nn.Module):
    """GATr + mean-pool head for R7 patch-injected binary jet classification.

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
        embedding : dict from embed_tagging_data_r7_patch
            "mv"       : Tensor (n_total, 1, 8)
            "patch_mv" : Tensor (n_total, K, 8)
            "batch"    : Tensor (n_total,) jet index per token
            "is_beam"  : Tensor (n_total,) bool (not used for masking here)

        Returns
        -------
        logits : Tensor (batchsize, 1)
        """
        mv = embedding["mv"].unsqueeze(0)              # (1, n_total, 1, 8)
        patch_mv = embedding["patch_mv"].unsqueeze(0)   # (1, n_total, K, 8)
        batch = embedding["batch"]                      # (n_total,)

        mask = build_attention_mask(batch, mv.dtype)

        mv_out, _ = self.net(mv, scalars=None, attention_mask=mask, patch_mv=patch_mv)
        scalars_out = mv_out[0, :, :, 0]  # (n_total, 1)

        batchsize = int(batch.max().item()) + 1
        logits = scatter(scalars_out, batch, dim=0, dim_size=batchsize, reduce="mean")
        return logits  # (batchsize, 1)
