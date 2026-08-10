"""fastfields.numpy: a friendly numpy interface over ``fastfields.dlpack``.

The underlying bindings operate *in place* / write through pre-allocated output
arrays. This package wraps them so every public function takes numpy arrays and
**returns** freshly allocated numpy arrays, never clobbering the caller's
input -- unless it is one of the trailing-underscore (``_``) variants, e.g.
``dt_euclidean_``/``sym_solve_``/``spline_coeff_``, which mutate their first
argument in place and return it. This mirrors the cupy backend's convention
exactly (``fastfields.cupy``'s docstring: "Trailing-underscore wrappers ...
operate in place"); torch omits the in-place spelling for ops whose backward
would need the pre-mutation value (see ``API_CONTRACT.md``, "In-place
policy").

Batch (leading) dimensions are **broadcast**: the raw bindings require every
input tensor of an op to share the same batch dims (they do not broadcast), so
these wrappers normalise inputs to a common broadcast batch shape and allocate
outputs with that shape. The broadcast is done **zero-copy**: each input is
re-strided to the target batch shape with 0-strides on the broadcast axes (a
view that shares memory with the original), which the stride-aware C++ library
consumes directly without any copy. Returned outputs therefore carry the
broadcast batch shape.

Read-only inputs are likewise passed with their native strides (no contiguous
copy is forced); only freshly allocated outputs and in-place (``_``-suffixed)
buffers are materialised as contiguous arrays.

Spline order and boundary condition arguments accept an ``int``, one of the
re-exported :class:`Spline` / :class:`Bound` enums, or a friendly string
(e.g. ``"cubic"``, ``"dct2"``).

The implementation is split by category into :mod:`._util` (validation and
batch-broadcast helpers), :mod:`._dt` (distance transforms), :mod:`._sym`
(compact-symmetric linear algebra) and :mod:`._resample` (spline coefficients
and resampling).
"""

from __future__ import annotations

from fastfields.dlpack import Bound, Spline

from ._dt import (
    dt_euclidean,
    dt_euclidean_,
    dt_l1,
    dt_l1_,
    dt_mesh,
    dt_spline_brent,
    dt_spline_gaussnewton,
    dt_spline_table,
)
from ._pushpull import count, grad, pull, push
from ._reg import (
    field_adddiag,
    field_adddiag_,
    field_addmatvec,
    field_addmatvec_,
    field_diag,
    field_diag_rls,
    field_forward,
    field_kernel,
    field_matvec,
    field_matvec_rls,
    field_precond,
    field_relax,
    field_relax_rls,
    field_subdiag,
    field_subdiag_,
    field_submatvec,
    field_submatvec_,
    flow_adddiag,
    flow_adddiag_,
    flow_addmatvec,
    flow_addmatvec_,
    flow_diag,
    flow_diag_rls,
    flow_forward,
    flow_kernel,
    flow_matvec,
    flow_matvec_rls,
    flow_precond,
    flow_relax,
    flow_relax_rls,
    flow_subdiag,
    flow_subdiag_,
    flow_submatvec,
    flow_submatvec_,
)
from ._resample import resample, restriction, spline_coeff, spline_coeff_
from ._sym import (
    sym_addmatvec_,
    sym_channels_from_packed,
    sym_invert,
    sym_invert_,
    sym_matvec,
    sym_matvec_backward,
    sym_solve,
    sym_solve_,
    sym_submatvec_,
)

__all__ = [
    "Spline",
    "Bound",
    "dt_euclidean",
    "dt_euclidean_",
    "dt_l1",
    "dt_l1_",
    "sym_matvec",
    "sym_matvec_backward",
    "sym_addmatvec_",
    "sym_submatvec_",
    "sym_solve",
    "sym_solve_",
    "sym_invert",
    "sym_invert_",
    "resample",
    "restriction",
    "spline_coeff",
    "spline_coeff_",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
    "dt_mesh",
    "sym_channels_from_packed",
    "pull",
    "push",
    "count",
    "grad",
    "field_matvec",
    "field_addmatvec",
    "field_addmatvec_",
    "field_submatvec",
    "field_submatvec_",
    "field_diag",
    "field_adddiag",
    "field_adddiag_",
    "field_subdiag",
    "field_subdiag_",
    "field_kernel",
    "field_relax",
    "field_matvec_rls",
    "field_diag_rls",
    "field_relax_rls",
    "field_precond",
    "field_forward",
    "flow_matvec",
    "flow_addmatvec",
    "flow_addmatvec_",
    "flow_submatvec",
    "flow_submatvec_",
    "flow_diag",
    "flow_adddiag",
    "flow_adddiag_",
    "flow_subdiag",
    "flow_subdiag_",
    "flow_kernel",
    "flow_relax",
    "flow_matvec_rls",
    "flow_diag_rls",
    "flow_relax_rls",
    "flow_precond",
    "flow_forward",
]
