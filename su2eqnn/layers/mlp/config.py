"""GeoMLP configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional


@dataclass
class MLPConfig:
    """Geometric MLP configuration.

    Parameters
    ----------
    mv_channels : list of int
        Multivector channel sizes per layer, from input to output.
        Must have at least 2 entries.
    s_channels : list of int or None
        Scalar channel sizes per layer. If provided, must have same length as mv_channels.
    activation : {"relu", "sigmoid", "gelu"}
        Gated activation function for nonlinear layers.
    dropout_prob : float or None
        If set, applies GradeDropout after the first (GeometricBilinear) layer.
    """

    mv_channels: Optional[List[int]] = None
    s_channels: Optional[List[int]] = None
    activation: str = "gelu"
    dropout_prob: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.dropout_prob, str) and self.dropout_prob.lower() in ("null", "none"):
            self.dropout_prob = None

    @classmethod
    def cast(cls, config: Any) -> MLPConfig:
        if isinstance(config, MLPConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Cannot cast {config} to {cls}")
