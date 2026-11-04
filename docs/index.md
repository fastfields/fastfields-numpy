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
| **Spline gather/scatter** | `pull`, `push`, `count`, `grad` |
| **Regularisers** | `field_*` (multi-channel field, per-channel penalties) and `flow_*` (vector flow, scalar penalties + the Lamé `shears`/`div` terms): the operator `*_matvec`, its diagonal `*_diag`, its stencil `*_kernel`, the `*_add…`/`*_sub…` accumulate forms of each, the RLS/JRLS-weighted `*_rls` variants, and the regularised system below |

Spline order and boundary arguments accept an `int`, a `Spline` / `Bound` enum,
or a friendly string like `"cubic"` or `"dct2"`.

### Solving the regularised system

With `H` the per-voxel data-term Hessian (compact-symmetric packing) and `L` a
regulariser:

| call | what it does |
|---|---|
| `field_forward` / `flow_forward` | apply `(H + L) x` |
| `field_precond` / `flow_precond` | apply `(H + diag(L))⁻¹ g` (Jacobi) |
| `field_precond_` / `flow_precond_` | the same, in place |
| `field_relax` / `flow_relax` | Gauss-Seidel sweeps on `(H + L) x = g` |
| `field_cg` | preconditioned conjugate gradients on the same system |

`field_cg` mutates `sol` in place and returns it, like `field_relax`; pass
`return_info=True` to also get the iteration count and final relative
residual. It is CPU-only in the underlying library (`fastfields.cupy` has no
`field_cg` for that reason).

## What this wrapper does not expose

Of the 58 operations in the underlying C++ hub, this package wraps all but
four: `pull_backward`, `push_backward`, `count_backward` and `grad_backward`.
Those are the adjoints of the gather/scatter operators with respect to both
the sampled field and the sampling coordinates, and they exist so an autograd
engine can differentiate through `grid` — which only `fastfields.torch` does.
There is no automatic differentiation on numpy, so they stay at the raw layer:
call `fastfields.dlpack.pull_backward(...)` if you need them.

`tests/test_api_coverage.py` fails if that list ever changes by accident.

See the [API reference](api/index.md) for full signatures and options.
