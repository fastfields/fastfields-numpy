# fastfields-numpy  (imports as `fastfields.numpy`)

A thin, user-friendly **numpy** interface over the `fastfields.dlpack`
bindings.

```
… ─ lib ─ dlpack ─ numpy ← (you are here) … ─ fastfields
```

## Philosophy / role
- The raw bindings operate **in place** / through pre-allocated outputs. This
  layer wraps them so every function accepts numpy arrays and **returns a fresh
  numpy array** (inputs untouched unless `inplace=True`).
- Adds validation, zero-copy batch broadcasting, and output allocation.
- Depends only on `fastfields-dlpack` + numpy.

## Exposed capabilities (feature level)
- **Distance**: `dt_euclidean`, `dt_l1` (along
  the last axis); point-to-spline (`dt_spline_{table,brent,gaussnewton}`)
  and point-to-mesh (`dt_mesh`).
- **Posdef** (compact-symmetric, packed trailing dim): `sym_matvec`,
  `sym_addmatvec_`/`sym_submatvec_`, `sym_matvec_backward`, `sym_solve`,
  `sym_invert`.
- **Resampling**: `spline_coeff`, `resample`, `restriction`.
- `Spline`/`Bound` enums re-exported; order/bound args accept int, enum, or a
  friendly string (`"cubic"`, `"dct2"`, …).

## Layout
`fastfields/numpy/`: `__init__.py` (public surface), `_dt.py` (distance),
`_sym.py` (posdef), `_resample.py` (resample/restriction/spline_coeff),
`_util.py` (validation/allocation helpers). `tests/test_numpy.py`.

## Build & test
```
pip install .                    # requires fastfields-dlpack installed
python -m pytest tests/ -q       # import from a neutral cwd
```
Prefer a regular install over editable (native-namespace merge; see caveats).

## Conventions & caveats
- **PEP 420 namespace**: ships only `fastfields/numpy/`, no
  `fastfields/__init__.py` — keeps `fastfields` a native namespace package.
- **Known caveat**: the point-to-spline / point-to-mesh distance ops have
  inconsistent shape contracts between their validation checks and kernels in
  the underlying `fastfields-cpu-lib` (can segfault / under-count). The wrappers
  follow the documented `distance.h` contract but these ops are **not** covered
  by the test suite. See MIGRATION.md.
- Ruff: line-length 79, select B/E/F/I/W (`pyproject.toml`).

## Pointers
- Hierarchy: `/home/user/.github/profile/README.md`.
- Status/caveats: `/home/user/fastfields-lib/MIGRATION.md`.
