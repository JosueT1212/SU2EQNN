"""Spin(3) ≅ SU(2) equivariant linear layer for Cl(3,0) multivectors.

Adapts L-GATr's EquiLinear to Cl(3,0):
  - 8 equivariant basis elements (4 grades × 2 multipliers) instead of 10
  - 8-dim multivectors instead of 16
  - pseudoscalar lives at index 7 (= -1), scalar at index 0 — same indexing trick works

The MIX_MVPSEUDOSCALAR_INTO_SCALAR coupling is valid here because e123 commutes
with all even (Spin(3)) elements, so mixing pseudoscalar ↔ scalar preserves
equivariance under Spin(3) (though it would break it under Pin(3)/O(3)).
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from su2eqnn.interface.scalar import embed_scalar
from su2eqnn.primitives.linear import equi_linear, NUM_SPIN3_LINEAR_BASIS_ELEMENTS

# Mix pseudoscalar (index 7) and scalar (index 0) with aux scalar stream.
# Valid under Spin(3) since e123 is central in Cl^0(3,0).
MIX_MVPSEUDOSCALAR_INTO_SCALAR = True

_N_BASIS = NUM_SPIN3_LINEAR_BASIS_ELEMENTS   # 8


class EquiLinear(nn.Module):
    """Spin(3)-equivariant linear layer for Cl(3,0) multivectors.

    Maps (multivectors, scalars) → (multivectors, scalars).

    multivectors: (..., in_mv_channels, 8) → (..., out_mv_channels, 8)
    scalars:      (..., in_s_channels)      → (..., out_s_channels)   [optional]

    Parameters
    ----------
    in_mv_channels  : int
    out_mv_channels : int
    in_s_channels   : int or None
    out_s_channels  : int or None
    bias            : bool
    initialization  : {"default", "small", "unit_scalar", "almost_unit_scalar"}
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
        bias: bool = True,
        initialization: str = "default",
    ) -> None:
        super().__init__()

        if initialization in ["unit_scalar", "almost_unit_scalar"]:
            assert bias, "unit_scalar initialization requires bias"
            if in_s_channels is None:
                raise NotImplementedError(
                    "unit_scalar initialization currently requires scalar inputs"
                )

        self._in_mv_channels = in_mv_channels

        # MV → MV weights: one coefficient per basis element
        self.weight = nn.Parameter(
            torch.empty(out_mv_channels, in_mv_channels, _N_BASIS)
        )

        # Scalar-only bias (grade-0 component only — other grades would break equivariance)
        self.bias = (
            nn.Parameter(torch.zeros(out_mv_channels, 1))
            if bias and in_s_channels is None
            else None
        )

        mix_factor = 2 if MIX_MVPSEUDOSCALAR_INTO_SCALAR else 1

        # Aux scalars → MV scalar (+ pseudoscalar if mixing)
        self.s2mvs: Optional[nn.Linear]
        if in_s_channels:
            self.s2mvs = nn.Linear(in_s_channels, mix_factor * out_mv_channels, bias=bias)
        else:
            self.s2mvs = None

        # MV scalar (+ pseudoscalar) → aux scalars
        self.mvs2s: Optional[nn.Linear]
        if out_s_channels:
            self.mvs2s = nn.Linear(mix_factor * in_mv_channels, out_s_channels, bias=bias)
        else:
            self.mvs2s = None

        # Aux scalars → aux scalars (skip connection)
        if in_s_channels is not None and out_s_channels is not None:
            self.s2s = nn.Linear(in_s_channels, out_s_channels, bias=False)
        else:
            self.s2s = None

        self.reset_parameters(initialization)

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Equivariant forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 8)
        scalars      : torch.Tensor with shape (..., in_s_channels) or None

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 8)
        outputs_s  : torch.Tensor with shape (..., out_s_channels) or None
        """
        outputs_mv = equi_linear(multivectors, self.weight)   # (..., out_channels, 8)

        if self.bias is not None:
            outputs_mv = outputs_mv + embed_scalar(self.bias)

        if self.s2mvs is not None and scalars is not None:
            if MIX_MVPSEUDOSCALAR_INTO_SCALAR:
                # indices 0 (scalar) and -1=7 (pseudoscalar) in 8-dim MV
                outputs_mv[..., [0, -1]] += self.s2mvs(scalars).view(
                    *outputs_mv.shape[:-2], outputs_mv.shape[-2], 2
                )
            else:
                outputs_mv[..., 0] += self.s2mvs(scalars)

        if self.mvs2s is not None:
            if MIX_MVPSEUDOSCALAR_INTO_SCALAR:
                outputs_s = self.mvs2s(
                    multivectors[..., [0, -1]].flatten(start_dim=-2)
                )
            else:
                outputs_s = self.mvs2s(multivectors[..., 0])
            if self.s2s is not None and scalars is not None:
                outputs_s = outputs_s + self.s2s(scalars)
        else:
            outputs_s = None

        return outputs_mv, outputs_s

    def reset_parameters(
        self,
        initialization: str,
        gain: float = 1.0,
        additional_factor: float = 1.0 / np.sqrt(3.0),
    ) -> None:
        mv_comp_factors, mv_factor, mvs_bias_shift, s_factor = self._compute_init_factors(
            initialization, gain, additional_factor
        )
        self._init_multivectors(mv_comp_factors, mv_factor, mvs_bias_shift)
        self._init_scalars(s_factor)

    @staticmethod
    def _compute_init_factors(initialization, gain, additional_factor):
        if initialization not in {"default", "small", "unit_scalar", "almost_unit_scalar"}:
            raise ValueError(f"Unknown initialization: {initialization}")

        if initialization == "default":
            mv_factor = gain * additional_factor * np.sqrt(3)
            s_factor  = gain * additional_factor * np.sqrt(3)
            mvs_bias_shift = 0.0
        elif initialization == "small":
            mv_factor = 0.1 * gain * additional_factor * np.sqrt(3)
            s_factor  = 0.1 * gain * additional_factor * np.sqrt(3)
            mvs_bias_shift = 0.0
        elif initialization == "unit_scalar":
            mv_factor = 0.1 * gain * additional_factor * np.sqrt(3)
            s_factor  = gain * additional_factor * np.sqrt(3)
            mvs_bias_shift = 1.0
        else:  # almost_unit_scalar
            mv_factor = 0.5 * gain * additional_factor * np.sqrt(3)
            s_factor  = gain * additional_factor * np.sqrt(3)
            mvs_bias_shift = 1.0

        mv_component_factors = torch.ones(_N_BASIS)  # 8 basis elements
        return mv_component_factors, mv_factor, mvs_bias_shift, s_factor

    def _init_multivectors(self, mv_component_factors, mv_factor, mvs_bias_shift):
        fan_in = self._in_mv_channels
        bound = mv_factor / np.sqrt(fan_in)
        for i, factor in enumerate(mv_component_factors):
            nn.init.uniform_(self.weight[..., i], a=-factor * bound, b=factor * bound)

        if self.s2mvs is not None:
            # Halve variance of scalar/pseudoscalar weights (shared contribution)
            bound = mv_component_factors[0] * mv_factor / np.sqrt(fan_in) / np.sqrt(2)
            nn.init.uniform_(self.weight[..., [0]], a=-bound, b=bound)
            if MIX_MVPSEUDOSCALAR_INTO_SCALAR:
                bound = mv_component_factors[-1] * mv_factor / np.sqrt(fan_in) / np.sqrt(2)
                nn.init.uniform_(self.weight[..., [-1]], a=-bound, b=bound)

            fan_in_s, _ = nn.init._calculate_fan_in_and_fan_out(self.s2mvs.weight)
            fan_in_s = max(fan_in_s, 1)
            bound = mv_component_factors[0] * mv_factor / np.sqrt(fan_in_s) / np.sqrt(2)
            nn.init.uniform_(self.s2mvs.weight, a=-bound, b=bound)

            if self.s2mvs.bias is not None:
                fan_combined = fan_in_s + self._in_mv_channels
                bound = mv_component_factors[0] / np.sqrt(fan_combined) if fan_combined > 0 else 0
                nn.init.uniform_(self.s2mvs.bias, mvs_bias_shift - bound, mvs_bias_shift + bound)

    def _init_scalars(self, s_factor):
        models = []
        if self.s2s:
            models.append(self.s2s)
        if self.mvs2s:
            models.append(self.mvs2s)
        for model in models:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.weight)
            fan_in = max(fan_in, 1)
            bound = s_factor / np.sqrt(fan_in) / np.sqrt(len(models))
            nn.init.uniform_(model.weight, a=-bound, b=bound)
        if self.mvs2s and self.mvs2s.bias is not None:
            fan_in = nn.init._calculate_fan_in_and_fan_out(self.mvs2s.weight)[0]
            if self.s2s:
                fan_in += nn.init._calculate_fan_in_and_fan_out(self.s2s.weight)[0]
            bound = s_factor / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.mvs2s.bias, -bound, bound)
