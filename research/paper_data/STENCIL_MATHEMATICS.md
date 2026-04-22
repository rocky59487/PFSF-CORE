# PFSF Mathematical Basis: Stencil Weights

## 1. Isotropic Laplacian Derivation
To ensure rotational invariance on a discrete 3D voxel grid, we adopt the **Shinozaki-Oono** stencil weights. This prevents "grid-aligned" fracture patterns and ensures stress flow correctly through diagonals.

| Connection | Distance | Relative Weight | Constant Value |
|---|---|---|---|
| **Face** | 1.0 | 1.0 | `1.0` |
| **Edge** | $\sqrt{2}$ | 0.5 | `EDGE_P = 0.5` |
| **Corner** | $\sqrt{3}$ | 1/6 | `CORNER_P = 0.1666667` |

## 2. Validation Scenarios
The following scenarios are used to prove "Continuum Alignment":
- **Scenario A: Cantilever Beam** (1D Analytic vs 3D Voxel)
- **Scenario B: Semi-Circular Arch** (Compression stability)
- **Scenario C: Deep Beam** (Shear-correction verification)
