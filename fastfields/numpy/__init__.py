"""fastfields.numpy: a friendly numpy interface over ``fastfields.dlpack``.

The underlying bindings operate *in place* / write through pre-allocated output
arrays. This package wraps them so every public function takes numpy arrays and
**returns** freshly allocated numpy arrays (never clobbering the caller's input
unless ``inplace=True`` is explicitly requested).

Batch (leading) dimensions are **broadcast**: the raw bindings require every
input tensor of an op to share the same batch dims (they do not broadcast), so
these wrappers normalise inputs to a common broadcast batch shape and allocate
outputs with that shape. The broadcast is done **zero-copy**: each input is
re-strided to the target batch shape with 0-strides on the broadcast axes (a
view that shares memory with the original), which the stride-aware C++ library
consumes directly without any copy. Returned outputs therefore carry the
broadcast batch shape.

Read-only inputs are likewise passed with their native strides (no contiguous
copy is forced); only freshly allocated outputs and functional (non-inplace)
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
    dt_l1,
    dt_mesh,
    dt_spline_brent,
    dt_spline_gaussnewton,
    dt_spline_table,
)
from ._pushpull import count, grad, pull, push
from ._reg import (
    field_diag,
    field_matvec,
    flow_diag,
    flow_diag_add,
    flow_diag_add_,
    flow_diag_sub,
    flow_diag_sub_,
    flow_forward,
    flow_kernel,
    flow_matvec,
    flow_matvec_add,
    flow_matvec_add_,
    flow_matvec_sub,
    flow_matvec_sub_,
    flow_precond,
    flow_relax,
)
from ._resample import resample, restriction, spline_coeff
from ._sym import (
    sym_addmatvec_,
    sym_channels_from_packed,
    sym_invert,
    sym_matvec,
    sym_matvec_backward,
    sym_solve,
    sym_submatvec_,
)

__all__ = [
    "Spline",
    "Bound",
    "dt_euclidean",
    "dt_l1",
    "sym_matvec",
    "sym_matvec_backward",
    "sym_addmatvec_",
    "sym_submatvec_",
    "sym_solve",
    "sym_invert",
    "resample",
    "restriction",
    "spline_coeff",
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
    "field_diag",
    "flow_matvec",
    "flow_matvec_add",
    "flow_matvec_add_",
    "flow_matvec_sub",
    "flow_matvec_sub_",
    "flow_diag",
    "flow_diag_add",
    "flow_diag_add_",
    "flow_diag_sub",
    "flow_diag_sub_",
    "flow_kernel",
    "flow_relax",
    "flow_precond",
    "flow_forward",
]
