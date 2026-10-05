"""SU2EQNN — Spin(3) ≅ SU(2) equivariant neural network layers on Cl(3,0).

Stepping stone toward a full Lorentz × Standard-Model-gauge equivariant
network (L-GATr + Furey internal symmetry). This package implements
the SU(2) internal-symmetry factor in isolation using Clifford algebra Cl(3,0).

Implemented so far (up to equivariant linear layer):
  su2eqnn.primitives  — bilinear (GP), linear basis, invariants, nonlinearities, norm
  su2eqnn.layers      — EquiLinear, EquiLayerNorm
  su2eqnn.interface   — embed/extract 3-vectors and scalars

Not yet implemented (pending research):
  - Gauge-invariant attention (inner product & encoding)
  - Geometric product layer (bilinear layer in CGENN style)
  - Full transformer block / SU2EQNN net
"""
