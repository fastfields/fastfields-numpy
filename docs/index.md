# fastfields-numpy

**fastfields-numpy** brings the fastfields field operators to plain **NumPy
arrays** on the CPU. Every function takes NumPy arrays in and returns fresh
NumPy arrays out — your inputs are left untouched unless you explicitly ask for
an in-place variant.

## Install

```sh
pip install fastfields-numpy \
    --extra-index-url https://fastfields.github.io/whl/cpu/
```

## Use it

```python
import numpy as np
import fastfields.numpy as ff

mask = np.zeros((256, 256), "float32")
mask[:, 128] = 1.0

dist = ff.dt_euclidean(mask)      # squared Euclidean distance along the last axis
```

## What's inside

| Operation | Functions |
|---|---|
| **Distance transforms** | `dt_euclidean`, `dt_l1` (along the last axis); point-to-spline `dt_spline_table` / `dt_spline_brent` / `dt_spline_gaussnewton`; point-to-mesh `dt_mesh` |
| **Positive-definite linear algebra** | `sym_matvec`, `sym_addmatvec_`, `sym_submatvec_`, `sym_solve`, `sym_invert` over whole fields of small symmetric matrices |
| **Resampling** | `resample` (spline up/down-sampling), `restriction` (its adjoint), `spline_coeff` (coefficient prefilter) |

Spline order and boundary arguments accept an `int`, a `Spline` / `Bound` enum,
or a friendly string like `"cubic"` or `"dct2"`.

See the [API reference](api/index.md) for full signatures and options.
