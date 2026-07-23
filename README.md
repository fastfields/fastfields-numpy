# fastfields-numpy

A thin, user-friendly numpy interface over the
[`fastfields.dlpack`](../fastfields-bind-py) nanobind bindings.

The bindings themselves operate *in place* / write through pre-allocated output
arrays.  `fastfields.numpy` wraps them so that every function accepts numpy
arrays and **returns** a freshly allocated numpy array (the input is never
clobbered unless you pass `inplace=True`).

```python
import numpy as np
import fastfields.numpy as ff

x = np.array([0, np.inf, np.inf, 0, np.inf], dtype=np.float32)
d = ff.dt_euclidean(x)      # squared EDT along the last axis
```

## Public API

Distance transforms (along the last axis; features are `0`, background `+inf`):

- `dt_euclidean(x, voxel_spacing=1.0, *, inplace=False)`
- `dt_l1(x, voxel_spacing=1.0, *, inplace=False)`

Compact-symmetric linear algebra (packed trailing dim `C*(C+1)/2`,
diagonal-first then upper rows; `C=2 -> [h00,h11,h01]`,
`C=3 -> [h00,h11,h22,h01,h02,h12]`):

- `sym_matvec(mat, vec)` -> `H @ vec`
- `sym_addmatvec_(out0, mat, vec)` / `sym_submatvec_(out0, mat, vec)`
- `sym_matvec_backward(grad, vec)`
- `sym_solve(mat, vec, weight=None)` -> `(H + diag(weight)) \ vec`
- `sym_invert(mat)` -> packed inverse

Spline coefficients & resampling:

- `spline_coeff(x, order=3, bound='dct2', *, inplace=False)`
- `resample(x, factor=None, shape=None, *, order=2, bound='dct2', ndim=None, shift=0.0)`
- `restriction(x, factor=None, shape=None, *, order=2, bound='dct2', ndim=None, shift=0.0)`

Point-to-spline / point-to-mesh distance (thin pass-throughs, see caveat below):

- `dt_spline_table(loc, coeff, times, *, order=3, bound='dct2')`
- `dt_spline_brent(loc, coeff, *, max_iter=64, tol=1e-6, step=0.1, ...)`
- `dt_spline_gaussnewton(loc, coeff, *, max_iter=64, tol=1e-6, ...)`
- `dt_mesh(loc, vertices, faces, *, signed=True, naive=False, return_nearest=False)`

`Spline` and `Bound` enums are re-exported.  Order/bound arguments accept an
`int`, an enum member, or a friendly string (`"cubic"`, `"dct2"`, ...).

## Caveat: spline/mesh distance ops

The point-to-spline and point-to-mesh distance ops in the underlying
`fastfields-cpu-lib` currently have inconsistent shape contracts between their
argument-validation checks and their kernels (they can segfault or under-count).
The wrappers here follow the documented `distance.h` contract, but these ops are
**not** covered by the test suite.  See the migration report for details.
