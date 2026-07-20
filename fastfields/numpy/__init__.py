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
    euclidean_distance_transform,
    l1_distance_transform,
    mesh_distance,
    spline_distance_brent,
    spline_distance_gaussnewton,
    spline_distance_table,
)
from ._resample import resample, restriction, spline_coeff
from ._sym import (
    sym_addmatvec,
    sym_channels_from_packed,
    sym_invert,
    sym_matvec,
    sym_matvec_backward,
    sym_solve,
    sym_submatvec,
)

__all__ = [
    "Spline",
    "Bound",
    "euclidean_distance_transform",
    "l1_distance_transform",
    "sym_matvec",
    "sym_matvec_backward",
    "sym_addmatvec",
    "sym_submatvec",
    "sym_solve",
    "sym_invert",
    "resample",
    "restriction",
    "spline_coeff",
    "spline_distance_table",
    "spline_distance_brent",
    "spline_distance_gaussnewton",
    "mesh_distance",
    "sym_channels_from_packed",
]
