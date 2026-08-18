"""Tests for fastfields_numpy (the friendly numpy wrappers)."""

import numpy as np
import pytest

import fastfields.numpy as ff

# --------------------------------------------------------------------------- #
# distance transforms                                                         #
# --------------------------------------------------------------------------- #


def _dt_reference(inp, voxel_spacing, cost):
    """Brute force: out[..., i] = min_j inp[..., j] + cost(spacing*(i-j))."""
    n = inp.shape[-1]
    flat = inp.reshape(-1, n)
    out = np.full_like(flat, np.inf)
    for r in range(flat.shape[0]):
        for i in range(n):
            best = np.inf
            for j in range(n):
                best = min(best, flat[r, j] + cost(voxel_spacing * (i - j)))
            out[r, i] = best
    return out.reshape(inp.shape)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_euclidean_matches_bruteforce(dtype):
    inp = np.array(
        [
            [0, np.inf, np.inf, 0, np.inf, np.inf, np.inf],
            [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf],
        ],
        dtype=dtype,
    )
    ref = _dt_reference(inp, 1.5, lambda d: d * d)
    out = ff.dt_euclidean(inp, voxel_spacing=1.5)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
    # input must be untouched (returns a new array)
    assert np.isinf(inp[0, 1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_l1_matches_bruteforce(dtype):
    inp = np.array(
        [
            [0, np.inf, np.inf, 0, np.inf, np.inf, np.inf],
            [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf],
        ],
        dtype=dtype,
    )
    ref = _dt_reference(inp, 2.0, lambda d: abs(d))
    out = ff.dt_l1(inp, voxel_spacing=2.0)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_l1_inplace_matches_out_of_place():
    inp = np.array([[0, np.inf, np.inf, 0, np.inf]], dtype=np.float64)
    expected = ff.dt_l1(inp, voxel_spacing=1.5)
    ret = ff.dt_l1_(inp, 1.5)
    assert ret is inp
    np.testing.assert_allclose(inp, expected, rtol=1e-10, atol=1e-10)


def test_dt_handles_noncontiguous_input():
    # Build a strided / transposed (non-C-contiguous) view; the wrapper must
    # copy it into a contiguous buffer and still produce the right answer.
    base = np.array(
        [
            [0, np.inf, np.inf, 0, np.inf, np.inf, np.inf],
            [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf],
        ],
        dtype=np.float64,
    )
    view = np.asfortranarray(base)  # F-contiguous, not C-contiguous
    assert not view.flags["C_CONTIGUOUS"]
    ref = _dt_reference(base, 1.0, lambda d: d * d)
    out = ff.dt_euclidean(view)
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)

    # A transposed view of a 3d array (last axis is the strided one).
    vol = np.stack([base, base + 0.0], axis=0)  # (2,2,7)
    tview = np.swapaxes(vol, 1, 2)  # (2,7,2), strided last
    assert not tview.flags["C_CONTIGUOUS"]
    ref_t = _dt_reference(tview.astype(np.float64), 1.0, lambda d: d * d)
    out_t = ff.dt_euclidean(tview)
    np.testing.assert_allclose(out_t, ref_t, rtol=1e-6, atol=1e-6)


def test_dt_inplace():
    inp = np.array([[0, np.inf, np.inf, 0, np.inf]], dtype=np.float32)
    ref = _dt_reference(inp, 1.0, lambda d: d * d)
    ret = ff.dt_euclidean_(inp)
    assert ret is inp
    np.testing.assert_allclose(inp, ref, rtol=1e-5, atol=1e-5)


def test_dt_mesh_signed_naive_return_nearest_are_not_keyword_only():
    # `signed`/`naive`/`return_nearest` must be positional-or-keyword here,
    # matching the torch/cupy wrappers -- an earlier revision made them
    # keyword-only on numpy only, so a positional call worked on two backends
    # and raised TypeError on the third (fastfields#4). This asserts the
    # Python-level signature directly: dt_mesh's own shape-checking layer
    # over the raw binding is separately known-unvalidated (see this
    # package's module docstring / CLAUDE.md), so a full native round-trip
    # isn't the right way to pin down a signature fix.
    import inspect

    sig = inspect.signature(ff.dt_mesh)
    for name in ("signed", "naive", "return_nearest"):
        assert sig.parameters[name].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ), f"{name} must not be keyword-only"


# --------------------------------------------------------------------------- #
# point-to-mesh distance (numerical, vs a brute-force reference)              #
# --------------------------------------------------------------------------- #
#
# One tetrahedron, many query points -- the ordinary `dt_mesh` call shape.
# Vertices/faces describe a *single* mesh and are never batched (only `loc`
# is); see fastfields#32.

# Unit tetrahedron, faces oriented outwards (needed for the signed variant,
# whose sign comes from the pseudo-normals of the nearest entity).
_TETRA_VERTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
_TETRA_FACES = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]])


def _closest_point_on_triangle(p, a, b, c):
    """Closest point to `p` on triangle `abc` (Ericson, RTCD 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denom = 1.0 / (va + vb + vc)
    return a + ab * (vb * denom) + ac * (vc * denom)


def _mesh_dt_reference(loc, verts, faces):
    """Brute force: per point, scan every triangle.

    Returns ``(dist, nearest_vertex)`` where ``dist`` is the *unsigned*
    Euclidean distance to the mesh surface and ``nearest_vertex`` is the
    vertex of the closest triangle that is nearest to the projection --
    the convention implemented by the kernel's ``get_nearest_vertex``.
    """
    flat = loc.reshape(-1, loc.shape[-1])
    dist = np.empty(len(flat))
    near = np.empty(len(flat), dtype=np.int64)
    for i, p in enumerate(flat):
        best_d, best_proj, best_face = np.inf, None, None
        for face in faces:
            q = _closest_point_on_triangle(p, *[verts[j] for j in face])
            d = np.linalg.norm(p - q)
            if d < best_d:
                best_d, best_proj, best_face = d, q, face
        dist[i] = best_d
        vd = [np.linalg.norm(verts[j] - best_proj) for j in best_face]
        near[i] = best_face[int(np.argmin(vd))]
    batch = loc.shape[:-1]
    return dist.reshape(batch), near.reshape(batch)


def _inside_convex(loc, verts, faces):
    """Inside test for a *convex* mesh with outward-oriented faces."""
    flat = loc.reshape(-1, loc.shape[-1])
    inside = np.ones(len(flat), dtype=bool)
    for face in faces:
        a, b, c = (verts[j] for j in face)
        n = np.cross(b - a, c - a)
        inside &= (flat - a) @ n <= 0
    return inside.reshape(loc.shape[:-1])


def _query_points(rng, n):
    """Random points around *and inside* the tetrahedron.

    The tetrahedron occupies a small fraction of its bounding box, so
    uniform box sampling alone would essentially never land inside it and
    the signed test would never see a negative distance. Barycentric
    samples are mixed in to guarantee interior points.

    Points very close to the surface are dropped: the *sign* is ambiguous
    there, and so is the nearest face when two are equidistant.
    """
    outer = rng.uniform(-0.6, 1.2, size=(3 * n, 3))
    inner = rng.dirichlet(np.ones(4), size=n) @ _TETRA_VERTS
    pts = np.concatenate([outer, inner])
    d, _ = _mesh_dt_reference(pts, _TETRA_VERTS, _TETRA_FACES)
    pts = pts[d > 1e-2]
    rng.shuffle(pts)
    return pts[:n]


@pytest.mark.parametrize("naive", [False, True])
def test_dt_mesh_unsigned_matches_bruteforce(naive):
    rng = np.random.default_rng(0)
    loc = _query_points(rng, 24)
    ref, _ = _mesh_dt_reference(loc, _TETRA_VERTS, _TETRA_FACES)

    out = ff.dt_mesh(
        loc, _TETRA_VERTS, _TETRA_FACES, signed=False, naive=naive
    )

    assert out.shape == loc.shape[:-1]
    assert out.dtype == loc.dtype
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("naive", [False, True])
def test_dt_mesh_signed_matches_bruteforce(naive):
    rng = np.random.default_rng(1)
    loc = _query_points(rng, 24)
    ref, _ = _mesh_dt_reference(loc, _TETRA_VERTS, _TETRA_FACES)
    sign = np.where(_inside_convex(loc, _TETRA_VERTS, _TETRA_FACES), -1.0, 1.0)

    out = ff.dt_mesh(loc, _TETRA_VERTS, _TETRA_FACES, signed=True, naive=naive)

    np.testing.assert_allclose(out, sign * ref, rtol=1e-10, atol=1e-10)
    # the tetrahedron is a closed surface: some points really are inside
    assert (sign < 0).any()


@pytest.mark.parametrize("signed", [False, True])
def test_dt_mesh_return_nearest_matches_bruteforce(signed):
    rng = np.random.default_rng(2)
    loc = _query_points(rng, 24)
    ref, ref_near = _mesh_dt_reference(loc, _TETRA_VERTS, _TETRA_FACES)
    if signed:
        ref = ref * np.where(
            _inside_convex(loc, _TETRA_VERTS, _TETRA_FACES), -1.0, 1.0
        )

    dist, near = ff.dt_mesh(
        loc, _TETRA_VERTS, _TETRA_FACES, signed=signed, return_nearest=True
    )

    assert near.shape == loc.shape[:-1]
    assert near.dtype == np.int64
    np.testing.assert_allclose(dist, ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(near, ref_near)


def test_dt_mesh_default_return_nearest_false_runs():
    # `return_nearest=False` is the default and passes a null `nearest_vertex`
    # down to the binding; that null used to be caught by the hub's
    # same-device check and reported as a (bogus) device mismatch --
    # fastfields#32.
    rng = np.random.default_rng(3)
    loc = _query_points(rng, 8)
    out = ff.dt_mesh(loc, _TETRA_VERTS, _TETRA_FACES)
    assert isinstance(out, np.ndarray)
    assert out.shape == loc.shape[:-1]


def test_dt_mesh_batched_query_points():
    # only `loc` carries batch dims; the outputs take loc.shape[:-1]
    rng = np.random.default_rng(4)
    loc = _query_points(rng, 12).reshape(3, 4, 3)
    ref, ref_near = _mesh_dt_reference(loc, _TETRA_VERTS, _TETRA_FACES)

    dist, near = ff.dt_mesh(
        loc, _TETRA_VERTS, _TETRA_FACES, signed=False, return_nearest=True
    )

    assert dist.shape == (3, 4)
    np.testing.assert_allclose(dist, ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(near, ref_near)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dt_mesh_dtypes(dtype):
    rng = np.random.default_rng(5)
    loc = _query_points(rng, 12)
    ref, _ = _mesh_dt_reference(loc, _TETRA_VERTS, _TETRA_FACES)
    out = ff.dt_mesh(
        loc.astype(dtype),
        _TETRA_VERTS.astype(dtype),
        _TETRA_FACES,
        signed=False,
    )
    assert out.dtype == dtype
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_dt_mesh_rejects_batched_mesh():
    # A batched mesh is not a supported shape (jitfields has no such mode
    # either); it must be rejected here with a clear message rather than
    # deep inside the native shape checks.
    rng = np.random.default_rng(6)
    loc = _query_points(rng, 5)
    with pytest.raises(ValueError, match="vertices must be a 2D"):
        ff.dt_mesh(loc, np.broadcast_to(_TETRA_VERTS, (5, 4, 3)), _TETRA_FACES)
    with pytest.raises(ValueError, match="faces must be a 2D"):
        ff.dt_mesh(loc, _TETRA_VERTS, np.broadcast_to(_TETRA_FACES, (5, 4, 3)))


# --------------------------------------------------------------------------- #
# compact-symmetric linear algebra                                            #
# --------------------------------------------------------------------------- #


def _pack_symmetric(mats):
    """Dense (B,C,C) symmetric -> compact (B, C*(C+1)/2) diagonal-then-rows."""
    B, C, _ = mats.shape
    packed = np.zeros((B, C * (C + 1) // 2), dtype=mats.dtype)
    for b in range(B):
        idx = 0
        for k in range(C):
            packed[b, idx] = mats[b, k, k]
            idx += 1
        for i in range(C):
            for j in range(i + 1, C):
                packed[b, idx] = mats[b, i, j]
                idx += 1
    return packed


def _random_symmetric(B, C, seed, posdef=False):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((B, C, C))
    m = m + np.transpose(m, (0, 2, 1))
    if posdef:
        m = np.einsum("bij,bkj->bik", m, m) + C * np.eye(C)[None]
    return m


@pytest.mark.parametrize("C", [2, 3])
def test_sym_matvec_matches_dense(C):
    B = 5
    mats = _random_symmetric(B, C, seed=C)
    vec = np.random.default_rng(100 + C).standard_normal((B, C))
    packed = _pack_symmetric(mats)

    out = ff.sym_matvec(packed, vec)
    ref = np.einsum("bij,bj->bi", mats, vec)
    np.testing.assert_allclose(out, ref, rtol=1e-8, atol=1e-8)
    assert out.shape == vec.shape


def test_sym_matvec_mixed_dtype_promotes_not_downcasts():
    # A float32 mat + float64 vec must promote to float64 (upcast the matrix),
    # never silently downcast the vector -- that would drop precision.
    C = 3
    mats = _random_symmetric(4, C, seed=7)
    packed = _pack_symmetric(mats).astype(np.float32)
    vec = np.random.default_rng(7).standard_normal((4, C)).astype(np.float64)

    out = ff.sym_matvec(packed, vec)
    assert out.dtype == np.float64
    # dense equivalent of the float32-rounded packing, promoted to float64;
    # the float64 vector keeps its precision through the product.
    mats32 = mats.astype(np.float32).astype(np.float64)
    ref = np.einsum("bij,bj->bi", mats32, vec)
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("C", [2, 3])
def test_sym_solve_inverts_matvec(C):
    B = 6
    mats = _random_symmetric(B, C, seed=C, posdef=True)
    packed = _pack_symmetric(mats)
    x = np.random.default_rng(7 + C).standard_normal((B, C))

    b = ff.sym_matvec(packed, x)
    x_rec = ff.sym_solve(packed, b)
    # the posdef solve accumulates in reduced precision, so loosen tolerance
    np.testing.assert_allclose(x_rec, x, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("C", [2, 3])
def test_sym_invert_then_matvec_is_identity(C):
    B = 4
    mats = _random_symmetric(B, C, seed=20 + C, posdef=True)
    packed = _pack_symmetric(mats)

    inv_packed = ff.sym_invert(packed)
    # (H^-1) applied to each column of H should give the identity.
    vecs = np.eye(C)[None].repeat(B, axis=0)  # (B, C, C): columns are e_k
    for k in range(C):
        col = np.ascontiguousarray(mats[:, :, k])  # H[:,k]
        rec = ff.sym_matvec(inv_packed, col)  # H^-1 @ H[:,k]
        np.testing.assert_allclose(rec, vecs[:, :, k], rtol=1e-5, atol=1e-5)


def test_sym_matvec_broadcasts_batch_dims():
    # hessian batch (1,) vs vec batch (5,): the wrapper must broadcast the
    # batch dims and produce the manually-broadcast dense result.
    C, B = 3, 5
    mats = _random_symmetric(1, C, seed=42)  # batch (1,)
    packed = _pack_symmetric(mats)  # (1, 6)
    vec = np.random.default_rng(7).standard_normal((B, C))  # batch (5,)

    out = ff.sym_matvec(packed, vec)
    assert out.shape == (B, C)
    dense = np.broadcast_to(mats, (B, C, C))
    ref = np.einsum("bij,bj->bi", dense, vec)
    np.testing.assert_allclose(out, ref, rtol=1e-8, atol=1e-8)


def test_sym_matvec_broadcast_is_zero_copy():
    # The big input must be broadcast as a 0-stride view (no copy). We probe
    # the wrapper's internal broadcast helper directly to assert zero-copy.
    from fastfields.numpy._util import _bcast_view

    big = np.ascontiguousarray(
        np.random.default_rng(0).standard_normal((1, 6))
    )
    view = _bcast_view(big, (100000, 6))
    assert view.shape == (100000, 6)
    assert view.strides[0] == 0  # broadcast axis has 0 stride
    assert np.shares_memory(view, big)  # no copy of the big input
    # end-to-end: mismatched batches still zero-copy on the larger operand
    packed = np.ascontiguousarray(
        _pack_symmetric(_random_symmetric(1, 2, seed=1))
    )
    vec = np.random.default_rng(2).standard_normal((256, 2))
    out = ff.sym_matvec(packed, vec)
    assert out.shape == (256, 2)


def test_sym_solve_broadcasts_weight():
    C, B = 3, 4
    mats = _random_symmetric(1, C, seed=99, posdef=True)  # batch (1,)
    packed = _pack_symmetric(mats)
    w = (
        np.abs(np.random.default_rng(1).standard_normal((B, C))) + 0.5
    )  # (4, C)
    x = np.random.default_rng(2).standard_normal((B, C))

    dense = np.broadcast_to(mats, (B, C, C))
    b = np.einsum("bij,bj->bi", dense, x) + w * x
    x_rec = ff.sym_solve(packed, b, weight=w)
    # the solve accumulates in reduced precision, so loosen tolerance
    np.testing.assert_allclose(x_rec, x, rtol=1e-4, atol=1e-4)


def test_sym_solve_with_weight():
    C, B = 3, 4
    mats = _random_symmetric(B, C, seed=99, posdef=True)
    packed = _pack_symmetric(mats)
    w = np.abs(np.random.default_rng(1).standard_normal((B, C))) + 0.5
    x = np.random.default_rng(2).standard_normal((B, C))

    # (H + diag(w)) x = b
    b = ff.sym_matvec(packed, x) + w * x
    x_rec = ff.sym_solve(packed, b, weight=w)
    np.testing.assert_allclose(x_rec, x, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("C", [2, 3])
def test_sym_solve_inplace_matches_out_of_place(C):
    # sym_solve_ mirrors the cupy backend: mutates the RHS in place and
    # returns it. It must agree numerically with the functional sym_solve.
    B = 5
    mats = _random_symmetric(B, C, seed=30 + C, posdef=True)
    packed = _pack_symmetric(mats)
    x = np.random.default_rng(9 + C).standard_normal((B, C))
    b = ff.sym_matvec(packed, x)

    expected = ff.sym_solve(packed, b)
    b_inplace = b.copy()
    ret = ff.sym_solve_(b_inplace, packed)
    assert ret is b_inplace
    np.testing.assert_allclose(b_inplace, expected, rtol=1e-6, atol=1e-6)


def test_sym_solve_inplace_broadcasts_weight():
    C, B = 3, 4
    mats = _random_symmetric(1, C, seed=77, posdef=True)  # batch (1,)
    packed = _pack_symmetric(mats)
    w = np.abs(np.random.default_rng(3).standard_normal((B, C))) + 0.5
    x = np.random.default_rng(4).standard_normal((B, C))

    dense = np.broadcast_to(mats, (B, C, C))
    b = np.einsum("bij,bj->bi", dense, x) + w * x
    ret = ff.sym_solve_(b, packed, weight=w)
    assert ret is b
    np.testing.assert_allclose(b, x, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("C", [2, 3])
def test_sym_invert_inplace_matches_out_of_place(C):
    B = 4
    mats = _random_symmetric(B, C, seed=40 + C, posdef=True)
    packed = _pack_symmetric(mats)

    expected = ff.sym_invert(packed)
    packed_inplace = packed.copy()
    ret = ff.sym_invert_(packed_inplace)
    assert ret is packed_inplace
    np.testing.assert_allclose(packed_inplace, expected, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# in-place add/sub matvec (C3: trailing-`_` must mutate the caller's array)   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("C", [2, 3])
def test_sym_addmatvec_mutates_caller_inplace(C):
    # Regression for fastfields-lib#17 (C3): the trailing-`_` variant must
    # write through the caller's accumulator, not return a private copy.
    B = 5
    mats = _random_symmetric(B, C, seed=C)
    packed = _pack_symmetric(mats)
    vec = np.random.default_rng(100 + C).standard_normal((B, C))
    out0 = np.random.default_rng(200 + C).standard_normal((B, C))

    expected = out0 + np.einsum("bij,bj->bi", mats, vec)
    ret = ff.sym_addmatvec_(out0, packed, vec)
    # returned object IS the caller's array, and it was updated in place.
    assert ret is out0
    np.testing.assert_allclose(out0, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("C", [2, 3])
def test_sym_submatvec_mutates_caller_inplace(C):
    B = 5
    mats = _random_symmetric(B, C, seed=C)
    packed = _pack_symmetric(mats)
    vec = np.random.default_rng(100 + C).standard_normal((B, C))
    out0 = np.random.default_rng(200 + C).standard_normal((B, C))

    expected = out0 - np.einsum("bij,bj->bi", mats, vec)
    ret = ff.sym_submatvec_(out0, packed, vec)
    assert ret is out0
    np.testing.assert_allclose(out0, expected, rtol=1e-8, atol=1e-8)


def test_sym_addmatvec_broadcasts_onto_accumulator():
    # out0 fixes the batch shape; mat/vec (batch (1,)) broadcast onto it.
    C, B = 3, 4
    mats = _random_symmetric(1, C, seed=11)  # batch (1,)
    packed = _pack_symmetric(mats)  # (1, 6)
    vec = np.random.default_rng(3).standard_normal((1, C))  # batch (1,)
    out0 = np.zeros((B, C))

    ff.sym_addmatvec_(out0, packed, vec)
    dense = np.broadcast_to(mats, (B, C, C))
    ref = np.broadcast_to(np.einsum("bij,bj->bi", dense, vec), (B, C))
    np.testing.assert_allclose(out0, ref, rtol=1e-8, atol=1e-8)


def test_sym_addmatvec_channel_mismatch_raises():
    packed = np.zeros(3)  # encodes C=2
    vec = np.zeros(3)  # C=3
    out0 = np.zeros(3)
    with pytest.raises(ValueError):
        ff.sym_addmatvec_(out0, packed, vec)


# --------------------------------------------------------------------------- #
# spline coeff / resample                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resample_factor_one_identity(dtype):
    x = np.array([1, 2, 3, 4, 5], dtype=dtype)
    out = ff.resample(x, factor=1, order="linear", bound="dct2")
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    np.testing.assert_allclose(out, x, rtol=1e-6, atol=1e-6)


def test_resample_no_args_identity_2d_last_axis():
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    out = ff.resample(x, order="linear", ndim=1)  # identity on last axis
    assert out.shape == x.shape
    np.testing.assert_allclose(out, x, rtol=1e-6, atol=1e-6)


def test_resample_upsample_shape_and_endpoints():
    x = np.arange(5, dtype=np.float64)
    out = ff.resample(x, factor=2, order="linear")
    # factor 2 -> 10 samples; align-corners keeps the endpoints and reproduces
    # the linear ramp over the same [0, 4] extent.
    assert out.shape == (10,)
    np.testing.assert_allclose(
        out, np.linspace(0, 4, 10), rtol=1e-6, atol=1e-6
    )


def test_resample_shape_arg():
    x = np.arange(4, dtype=np.float64)
    out = ff.resample(x, shape=7, order="linear")
    assert out.shape == (7,)
    np.testing.assert_allclose(out, np.linspace(0, 3, 7), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spline_coeff_shape_dtype_preserved(dtype):
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=dtype)
    out = ff.spline_coeff(x, order=3, bound="dct2")
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    # input not modified
    np.testing.assert_allclose(x, [[1, 2, 3, 4, 5]])


def test_spline_coeff_order1_noop():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    out = ff.spline_coeff(x, order=1)
    np.testing.assert_allclose(out, x, rtol=1e-12, atol=1e-12)


def test_spline_coeff_inplace_matches_out_of_place():
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    expected = ff.spline_coeff(x, order=3, bound="dct2")
    ret = ff.spline_coeff_(x, order=3, bound="dct2")
    assert ret is x
    np.testing.assert_allclose(x, expected, rtol=1e-12, atol=1e-12)


def test_restriction_runs_and_shapes():
    x = np.arange(9, dtype=np.float64)
    out = ff.restriction(x, shape=5, order="linear")
    assert out.shape == (5,)
    assert out.dtype == x.dtype


# --------------------------------------------------------------------------- #
# anchor conventions (match interpol.resize)                                  #
# --------------------------------------------------------------------------- #


def test_anchor_scale_shift_mapping():
    from fastfields.helpers import anchor_scale_shift as _anchor_scale_shift

    # 8 -> 4 downsample; scale/shift per torch-interpol convention.
    for name, abbr, exp_scale, exp_shift in [
        ("centers", "c", 7 / 3, 0.0),
        ("edges", "e", 2.0, 0.5),
        ("first", "f", 2.0, 0.0),
        ("last", "l", 2.0, 1.0),
    ]:
        scale, shift = _anchor_scale_shift(name, (8,), (4,), 1)
        assert shift == exp_shift
        np.testing.assert_allclose(scale, [exp_scale])
        # the abbreviation resolves to the same mapping
        assert _anchor_scale_shift(abbr, (8,), (4,), 1) == (scale, shift)


def test_anchor_unknown_raises():
    from fastfields.helpers import anchor_scale_shift as _anchor_scale_shift

    with pytest.raises(ValueError, match="anchor"):
        _anchor_scale_shift("nope", (8,), (4,), 1)
    with pytest.raises(ValueError, match="anchor"):
        ff.resample(np.arange(8, dtype=np.float64), shape=4, anchor="nope")


@pytest.mark.parametrize(
    "anchor,expected",
    [
        # linear interp of the ramp arange(8) reproduces the sampled
        # input-coordinate; all coords below stay inside [0, 7].
        ("centers", np.linspace(0, 7, 4)),
        ("first", [0.0, 2.0, 4.0, 6.0]),
        ("edges", [0.5, 2.5, 4.5, 6.5]),
        ("last", [1.0, 3.0, 5.0, 7.0]),
    ],
)
def test_resample_anchor_matches_grid(anchor, expected):
    x = np.arange(8, dtype=np.float64)
    out = ff.resample(x, shape=4, order="linear", anchor=anchor)
    assert out.shape == (4,)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_resample_default_anchor_is_centers():
    x = np.arange(8, dtype=np.float64)
    default = ff.resample(x, shape=4, order="linear")
    centers = ff.resample(x, shape=4, order="linear", anchor="centers")
    np.testing.assert_array_equal(default, centers)


def test_resample_scale_overrides_anchor():
    # An explicit scale overrides the anchor-derived scale; scale=in/out with
    # shift=0 reproduces the 'first' grid regardless of the anchor.
    x = np.arange(8, dtype=np.float64)
    override = ff.resample(
        x, shape=4, order="linear", anchor="centers", scale=[2.0], shift=0.0
    )
    first = ff.resample(x, shape=4, order="linear", anchor="first")
    np.testing.assert_allclose(override, first, rtol=1e-6, atol=1e-6)


def test_resample_scale_wrong_length_raises():
    x = np.arange(8, dtype=np.float64)
    with pytest.raises(ValueError, match="scale"):
        ff.resample(x, shape=4, ndim=1, scale=[2.0, 2.0])


def test_resample_shift_overrides_anchor():
    x = np.arange(8, dtype=np.float64)
    # explicit shift=0 turns 'last' (shift 1) into the 'first' grid (shift 0),
    # since both use the in/out scale.
    override = ff.resample(
        x, shape=4, order="linear", anchor="last", shift=0.0
    )
    first = ff.resample(x, shape=4, order="linear", anchor="first")
    np.testing.assert_allclose(override, first, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# argument normalisation                                                      #
# --------------------------------------------------------------------------- #


def test_enum_and_string_args_equivalent():
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    a = ff.spline_coeff(x, order=ff.Spline.Cubic, bound=ff.Bound.DCT2)
    b = ff.spline_coeff(x, order="cubic", bound="dct2")
    c = ff.spline_coeff(x, order=3, bound=3)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, c)


def test_bad_spline_and_bound_raise():
    x = np.zeros(4, dtype=np.float64)
    with pytest.raises(ValueError):
        ff.spline_coeff(x, order="nonsense")
    with pytest.raises(ValueError):
        ff.spline_coeff(x, bound="nonsense")


def test_channels_from_packed():
    assert ff.sym_channels_from_packed(3) == 2
    assert ff.sym_channels_from_packed(6) == 3
    with pytest.raises(ValueError):
        ff.sym_channels_from_packed(4)


# ------------------------------------------------------------------------- #
# in-place on non-contiguous arrays (library is stride-aware -> zero copy)   #
# ------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inplace_noncontiguous(dtype):
    # A strided (non-contiguous) view of the last axis must be writable in
    # place: the write lands in the caller's buffer, no contiguous copy.
    base = np.full((3, 16), np.inf, dtype=dtype)
    base[:, 0] = 0.0
    base[:, 8] = 0.0
    x = base[:, ::2]  # shape (3, 8), stride 2 (non-contig)
    assert not x.flags["C_CONTIGUOUS"]
    ref = np.ascontiguousarray(x)
    ff.dt_euclidean_(ref)
    ret = ff.dt_euclidean_(x)
    assert ret is x  # returned the same object
    assert np.allclose(x, ref)  # correct result on the strided view
    assert np.allclose(base[:, ::2], ref)  # write landed in the parent buffer


def test_inplace_rejects_non_array_and_bad_dtype():
    with pytest.raises(TypeError):
        ff.dt_euclidean_([0.0, 1.0])  # not ndarray
    with pytest.raises(TypeError):
        ff.dt_euclidean_(np.zeros(4, dtype=np.int32))  # wrong dtype


# --------------------------------------------------------------------------- #
# pushpull (spline gather / scatter)                                          #
# --------------------------------------------------------------------------- #


def test_pull_linear_interpolation():
    inp = np.array([[0.0], [10.0], [20.0], [30.0]])  # (4, 1) ramp
    grid = np.array([[0.5], [1.5], [2.5]])  # (3, 1) between voxels
    out = ff.pull(inp, grid, order=1)
    np.testing.assert_allclose(out[:, 0], [5.0, 15.0, 25.0])


def test_count_identity_is_ones():
    grid = np.arange(5.0).reshape(5, 1)
    np.testing.assert_allclose(ff.count(grid, shape=5, order=1)[:, 0], 1.0)


def test_push_is_pull_adjoint():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 1))
    y = rng.standard_normal((4, 1))
    grid = np.linspace(0, 5, 4).reshape(4, 1)
    px = ff.pull(x, grid, order=2)
    py = ff.push(y, grid, shape=6, order=2)
    np.testing.assert_allclose(
        float((px * y).sum()), float((x * py).sum()), rtol=1e-8, atol=1e-8
    )


def test_grad_of_ramp_is_constant_slope():
    inp = np.array([[0.0], [10.0], [20.0], [30.0]])
    grid = np.array([[0.5], [1.5], [2.5]])
    g = ff.grad(inp, grid, order=1)  # (3, 1, 1)
    np.testing.assert_allclose(g[:, 0, 0], 10.0)


# --------------------------------------------------------------------------- #
# regularisers                                                                #
# --------------------------------------------------------------------------- #


def test_field_matvec_absolute_is_per_channel_scaling():
    rng = np.random.default_rng(1)
    inp = rng.standard_normal((8, 2))
    out = ff.field_matvec(inp, absolute=[2.0, 3.0], ndim=1)
    np.testing.assert_allclose(out[:, 0], 2.0 * inp[:, 0])
    np.testing.assert_allclose(out[:, 1], 3.0 * inp[:, 1])


def test_field_diag_absolute():
    d = ff.field_diag((8, 2), absolute=2.0, ndim=1)
    np.testing.assert_allclose(d, 2.0)


def test_flow_matvec_absolute_is_scaling():
    rng = np.random.default_rng(2)
    inp = rng.standard_normal((8, 1))
    np.testing.assert_allclose(
        ff.flow_matvec(inp, absolute=2.5, ndim=1), 2.5 * inp
    )


def test_field_penalty_wrong_length_raises():
    with pytest.raises(ValueError):
        ff.field_matvec(np.zeros((4, 2)), absolute=[1.0, 2.0, 3.0], ndim=1)


@pytest.mark.parametrize(
    "kw",
    [
        {"membrane": 1.0},
        {"shears": 1.0, "div": 0.5},
        {
            "absolute": 0.3,
            "membrane": 0.7,
            "bending": 0.4,
            "shears": 1.0,
            "div": 0.5,
        },
    ],
)
def test_flow_relax_solves_system(kw):
    # relaxation drives (H + L) x -> g; with a strong diagonal Hessian the
    # Gauss-Seidel sweeps converge. Residual recomputes L x via flow_matvec.
    rng = np.random.default_rng(4)
    H, W, hdiag = 6, 7, 6.0
    hes = np.zeros((H, W, 3))
    hes[..., 0] = hdiag
    hes[..., 1] = hdiag
    grd = rng.standard_normal((H, W, 2))
    x = ff.flow_relax(np.zeros((H, W, 2)), hes, grd, ndim=2, nb_iter=150, **kw)
    lx = ff.flow_matvec(x, ndim=2, **kw)
    rel = np.linalg.norm(hdiag * x + lx - grd) / np.linalg.norm(grd)
    assert rel < 3e-3


@pytest.mark.parametrize("bound", ["dct2", "dft"])
@pytest.mark.parametrize(
    "kw",
    [
        {"shears": 1.0},
        {"div": 1.0},
        {
            "absolute": 0.3,
            "membrane": 0.5,
            "bending": 0.4,
            "shears": 1.3,
            "div": 0.7,
        },
    ],
)
def test_flow_matvec_lame_is_self_adjoint(kw, bound):
    # The linear-elastic (shears/div) flow operator must be self-adjoint
    # (SPD) under every boundary, including the reflecting DCT2 case.
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5, 6, 2))
    y = rng.standard_normal((5, 6, 2))
    lx = ff.flow_matvec(x, ndim=2, bound=bound, **kw)
    ly = ff.flow_matvec(y, ndim=2, bound=bound, **kw)
    np.testing.assert_allclose((lx * y).sum(), (x * ly).sum(), rtol=1e-6)


def _flow_hessian_2d(H, W, seed):
    """Per-voxel SPD 2x2 Hessian, packed compact-symmetric -> (H, W, 3)."""
    mats = _random_symmetric(H * W, 2, seed, posdef=True)
    return _pack_symmetric(mats).reshape(H, W, 3)


@pytest.mark.parametrize(
    "kw,is_matrix,width",
    [
        ({"absolute": 2.5}, False, 1),
        ({"membrane": 1.0}, False, 3),
        ({"bending": 1.0}, False, 5),
        ({"shears": 1.3, "div": 0.7}, True, 3),
        (
            {
                "absolute": 0.3,
                "membrane": 0.5,
                "bending": 0.4,
                "shears": 1.3,
                "div": 0.7,
            },
            True,
            5,
        ),
    ],
)
def test_flow_kernel_is_matvec_impulse_response(kw, is_matrix, width):
    # The materialised stencil equals flow_matvec's impulse response in the
    # interior (translation-invariant there).
    C = 2
    K = ff.flow_kernel(2, **kw)
    assert K.shape == (
        (width, width, C, C) if is_matrix else (width, width, C)
    )
    kd = width
    N, cc, half = 2 * kd + 1, kd, kd // 2
    for j0 in range(C):
        x = np.zeros((N, N, C))
        x[cc, cc, j0] = 1.0
        o = ff.flow_matvec(x, ndim=2, **kw)
        for a in range(kd):
            for b in range(kd):
                for i in range(C):
                    got = o[cc + a - half, cc + b - half, i]
                    kern = (
                        K[a, b, i, j0]
                        if is_matrix
                        else (K[a, b, i] if i == j0 else 0.0)
                    )
                    np.testing.assert_allclose(got, kern, atol=1e-10)


def test_flow_forward_is_sym_matvec_plus_flow_matvec():
    # (M + R) v == M v + R v, by construction.
    rng = np.random.default_rng(11)
    H, W = 5, 6
    mat = _flow_hessian_2d(H, W, 11)
    vec = rng.standard_normal((H, W, 2))
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5)
    fwd = ff.flow_forward(mat, vec, ndim=2, **kw)
    expect = ff.sym_matvec(mat, vec) + ff.flow_matvec(vec, ndim=2, **kw)
    np.testing.assert_allclose(fwd, expect, rtol=1e-6, atol=1e-6)


def test_flow_precond_solves_diagonal_system():
    # x = (M + diag(R)) \ v  =>  M x + diag(R) x == v.
    rng = np.random.default_rng(12)
    H, W = 5, 6
    mat = _flow_hessian_2d(H, W, 12)
    vec = rng.standard_normal((H, W, 2))
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5)
    x = ff.flow_precond(mat, vec, ndim=2, **kw)
    diag = ff.flow_diag(vec.shape, ndim=2, **kw)
    residual = ff.sym_matvec(mat, x) + diag * x - vec
    np.testing.assert_allclose(residual, 0.0, atol=1e-5)


def test_flow_matvec_accumulate_variants():
    rng = np.random.default_rng(21)
    H, W = 5, 6
    flow = rng.standard_normal((H, W, 2))
    base = rng.standard_normal((H, W, 2))
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5, ndim=2)
    L = ff.flow_matvec(flow, **kw)
    # fresh-array forms
    np.testing.assert_allclose(ff.flow_addmatvec(base, flow, **kw), base + L)
    np.testing.assert_allclose(ff.flow_submatvec(base, flow, **kw), base - L)
    # in-place forms mutate and return the same array
    a = base.copy()
    r = ff.flow_addmatvec_(a, flow, **kw)
    assert r is a
    np.testing.assert_allclose(a, base + L)
    s = base.copy()
    r = ff.flow_submatvec_(s, flow, **kw)
    assert r is s
    np.testing.assert_allclose(s, base - L)


def test_flow_diag_accumulate_variants():
    rng = np.random.default_rng(22)
    H, W = 5, 6
    base = rng.standard_normal((H, W, 2))
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5, ndim=2)
    d = ff.flow_diag(base.shape, **kw)
    np.testing.assert_allclose(ff.flow_adddiag(base, **kw), base + d)
    np.testing.assert_allclose(ff.flow_subdiag(base, **kw), base - d)
    a = base.copy()
    assert ff.flow_adddiag_(a, **kw) is a
    np.testing.assert_allclose(a, base + d)
    s = base.copy()
    assert ff.flow_subdiag_(s, **kw) is s
    np.testing.assert_allclose(s, base - d)


def _field_hessian(shape_spatial, C, seed):
    """Per-voxel SPD C×C Hessian, packed compact-symmetric."""
    n = int(np.prod(shape_spatial))
    mats = _random_symmetric(n, C, seed, posdef=True)
    packed = _pack_symmetric(mats)
    return packed.reshape(*shape_spatial, C * (C + 1) // 2)


def test_field_forward_is_sym_matvec_plus_field_matvec():
    rng = np.random.default_rng(31)
    H, W, C = 5, 6, 2
    mat = _field_hessian((H, W), C, 31)
    vec = rng.standard_normal((H, W, C))
    kw = dict(absolute=[0.3, 0.4], membrane=[0.7, 0.5], ndim=2)
    fwd = ff.field_forward(mat, vec, **kw)
    expect = ff.sym_matvec(mat, vec) + ff.field_matvec(vec, **kw)
    np.testing.assert_allclose(fwd, expect, rtol=1e-6, atol=1e-6)


def test_field_precond_solves_diagonal_system():
    rng = np.random.default_rng(32)
    H, W, C = 5, 6, 2
    mat = _field_hessian((H, W), C, 32)
    vec = rng.standard_normal((H, W, C))
    kw = dict(absolute=[0.3, 0.4], membrane=[0.7, 0.5], ndim=2)
    x = ff.field_precond(mat, vec, **kw)
    diag = ff.field_diag(vec.shape, **kw)
    residual = ff.sym_matvec(mat, x) + diag * x - vec
    np.testing.assert_allclose(residual, 0.0, atol=1e-5)


@pytest.mark.parametrize(
    "kw",
    [
        {"membrane": [1.0, 0.8]},
        {"absolute": [0.3, 0.4], "membrane": [0.7, 0.5]},
        {
            "absolute": [0.3, 0.4],
            "membrane": [0.7, 0.5],
            "bending": [0.4, 0.2],
        },
    ],
)
def test_field_relax_solves_system(kw):
    # relaxation drives (H + L) x -> g; with a strong diagonal Hessian the
    # Gauss-Seidel sweeps converge. Residual recomputes L x via field_matvec,
    # mirroring test_flow_relax_solves_system.
    rng = np.random.default_rng(34)
    H, W, C, hdiag = 6, 7, 2, 6.0
    hes = np.zeros((H, W, C * (C + 1) // 2))
    hes[..., 0] = hdiag
    hes[..., 1] = hdiag
    grd = rng.standard_normal((H, W, C))
    x = ff.field_relax(
        np.zeros((H, W, C)), hes, grd, ndim=2, nb_iter=250, **kw
    )
    lx = ff.field_matvec(x, ndim=2, **kw)
    rel = np.linalg.norm(hdiag * x + lx - grd) / np.linalg.norm(grd)
    assert rel < 3e-3


def test_field_relax_is_in_place():
    # `field_relax` mutates and returns its first argument, as flow_relax does.
    sol = np.zeros((6, 6, 2))
    hes = _field_hessian((6, 6), 2, 35)
    grd = np.ones((6, 6, 2))
    out = ff.field_relax(sol, hes, grd, membrane=1.0, ndim=2, nb_iter=4)
    assert out is sol
    assert np.any(sol != 0.0)


def test_field_accumulate_variants():
    rng = np.random.default_rng(33)
    H, W, C = 5, 6, 2
    field = rng.standard_normal((H, W, C))
    base = rng.standard_normal((H, W, C))
    kw = dict(absolute=[0.3, 0.4], membrane=[0.7, 0.5], ndim=2)
    L = ff.field_matvec(field, **kw)
    d = ff.field_diag(base.shape, **kw)
    np.testing.assert_allclose(ff.field_addmatvec(base, field, **kw), base + L)
    np.testing.assert_allclose(ff.field_submatvec(base, field, **kw), base - L)
    np.testing.assert_allclose(ff.field_adddiag(base, **kw), base + d)
    np.testing.assert_allclose(ff.field_subdiag(base, **kw), base - d)
    a = base.copy()
    assert ff.field_addmatvec_(a, field, **kw) is a
    np.testing.assert_allclose(a, base + L)
    s = base.copy()
    assert ff.field_subdiag_(s, **kw) is s
    np.testing.assert_allclose(s, base - d)


@pytest.mark.parametrize(
    "order,width,kw",
    [
        (1, 1, dict(absolute=[2.5, 1.5])),
        (2, 3, dict(absolute=[0.3, 0.4], membrane=[1.0, 0.7])),
        (
            3,
            5,
            dict(absolute=[0.3, 0.4], membrane=[0.5, 0.6], bending=[1.0, 0.8]),
        ),
    ],
)
def test_field_kernel_is_matvec_impulse_response(order, width, kw):
    # The per-channel field stencil equals field_matvec's impulse response in
    # the interior (channels are independent).
    C = 2
    K = ff.field_kernel(2, **kw)
    assert K.shape == (width, width, C)
    kd = width
    N, cc, half = 2 * kd + 1, kd, kd // 2
    for c0 in range(C):
        x = np.zeros((N, N, C))
        x[cc, cc, c0] = 1.0
        o = ff.field_matvec(x, ndim=2, **kw)
        for a in range(kd):
            for b in range(kd):
                for c in range(C):
                    got = o[cc + a - half, cc + b - half, c]
                    kern = K[a, b, c] if c == c0 else 0.0
                    np.testing.assert_allclose(got, kern, atol=1e-10)


def test_field_kernel_channels_from_penalty_length():
    # C is inferred from the per-channel penalty length; `channels` overrides.
    assert ff.field_kernel(2, absolute=[1.0, 2.0, 3.0]).shape == (1, 1, 3)
    assert ff.field_kernel(1, absolute=2.0, channels=4).shape == (1, 4)
    assert ff.field_kernel(2, membrane=[1.0]).shape == (3, 3, 1)


# --------------------------------------------------------------------------- #
# RLS/JRLS-weighted field regulariser                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kw",
    [
        {"absolute": [1.75, 0.9]},
        {"absolute": [0.3, 0.4], "membrane": [1.0, 0.7]},
        {
            "absolute": [0.3, 0.4],
            "membrane": [0.5, 0.6],
            "bending": [1.0, 0.8],
        },
    ],
)
def test_field_matvec_rls_unit_weight_matches_field_matvec(kw):
    # An all-ones weight map degenerates the weighted operator to the plain
    # one, for both the JRLS (wc=1) and RLS (wc=C) weight shapes.
    rng = np.random.default_rng(41)
    H, W, C = 6, 7, 2
    x = rng.standard_normal((H, W, C))
    expect = ff.field_matvec(x, ndim=2, **kw)
    for wc in (1, C):
        wgt = np.ones((H, W, wc))
        got = ff.field_matvec_rls(x, wgt, ndim=2, **kw)
        np.testing.assert_allclose(got, expect, atol=1e-10)


def test_field_diag_rls_matches_matvec_rls_on_impulse():
    rng = np.random.default_rng(42)
    H, W, C = 6, 7, 2
    kw = dict(absolute=[0.3, 0.4], membrane=[1.0, 0.7], ndim=2)
    wgt = 0.5 + np.abs(rng.standard_normal((H, W, 1)))
    d = ff.field_diag_rls(wgt, **kw)
    assert d.shape == (H, W, C)
    i, j, c = 3, 3, 1
    e = np.zeros((H, W, C))
    e[i, j, c] = 1.0
    o = ff.field_matvec_rls(e, wgt, **kw)
    np.testing.assert_allclose(d[i, j, c], o[i, j, c], atol=1e-8)


def test_field_matvec_rls_jrls_per_channel_decomposes():
    # The field regulariser never couples channels, so a genuine per-channel
    # weight map (wc == C) must decompose into C independent single-channel
    # problems, each solvable through the wc == 1 path -- an independent
    # ground truth, mirroring fastfields-cpu-lib's
    # run_2d_rls_jrls_per_channel (regression oracle for cpu-lib#65).
    rng = np.random.default_rng(43)
    H, W, C = 6, 7, 2
    kw = dict(
        absolute=[0.3, 0.4], membrane=[1.0, 0.7], bending=[0.5, 0.6], ndim=2
    )
    x = rng.standard_normal((H, W, C))
    wgt = 0.5 + np.abs(rng.standard_normal((H, W, C)))
    Lx = ff.field_matvec_rls(x, wgt, **kw)
    for c in range(C):
        kwc = {k: [v[c]] for k, v in kw.items() if k != "ndim"}
        kwc["ndim"] = 2
        xc = x[..., c : c + 1]
        wc = wgt[..., c : c + 1]
        Lc = ff.field_matvec_rls(xc, wc, **kwc)
        np.testing.assert_allclose(Lx[..., c : c + 1], Lc, atol=1e-8)


def test_field_relax_rls_solves_system():
    # Mirrors test_field_relax_solves_system, through the weighted operator
    # with an all-ones weight map (reduces to the unweighted system).
    rng = np.random.default_rng(44)
    H, W, C, hdiag = 6, 7, 2, 6.0
    kw = dict(absolute=[0.3, 0.4], membrane=[0.7, 0.5], ndim=2)
    hes = np.zeros((H, W, C * (C + 1) // 2))
    hes[..., 0] = hdiag
    hes[..., 1] = hdiag
    grd = rng.standard_normal((H, W, C))
    wgt = np.ones((H, W, 1))
    x = ff.field_relax_rls(
        np.zeros((H, W, C)), hes, grd, wgt, nb_iter=250, **kw
    )
    lx = ff.field_matvec_rls(x, wgt, **kw)
    rel = np.linalg.norm(hdiag * x + lx - grd) / np.linalg.norm(grd)
    assert rel < 3e-3


def test_field_relax_rls_is_in_place():
    sol = np.zeros((6, 6, 2))
    hes = _field_hessian((6, 6), 2, 45)
    grd = np.ones((6, 6, 2))
    wgt = np.ones((6, 6, 1))
    out = ff.field_relax_rls(
        sol, hes, grd, wgt, membrane=1.0, ndim=2, nb_iter=4
    )
    assert out is sol
    assert np.any(sol != 0.0)


# --------------------------------------------------------------------------- #
# Flow RLS/JRLS                                                               #
#                                                                             #
# The flow weighting is always *joint*: the trailing axis holds the           #
# components of one displacement vector, so `wgt` has a trailing size-1 axis  #
# (there is no per-channel RLS mode as on the field side). `bending` has no   #
# weighted kernel and is rejected.                                            #
# --------------------------------------------------------------------------- #

_FLOW_RLS_KW = [
    {"absolute": 1.75},
    {"absolute": 0.3, "membrane": 1.0},
    {"shears": 1.3, "div": 0.7},
    {"absolute": 0.5, "membrane": 0.9, "shears": 1.3, "div": 0.7},
]


@pytest.mark.parametrize("kw", _FLOW_RLS_KW)
def test_flow_matvec_rls_unit_weight_matches_flow_matvec(kw):
    # An all-ones weight map degenerates the weighted operator to the plain
    # one -- the same oracle used for field_matvec_rls.
    rng = np.random.default_rng(51)
    H, W, D = 6, 7, 2
    x = rng.standard_normal((H, W, D))
    expect = ff.flow_matvec(x, ndim=2, **kw)
    got = ff.flow_matvec_rls(x, np.ones((H, W, 1)), ndim=2, **kw)
    np.testing.assert_allclose(got, expect, atol=1e-10)


@pytest.mark.parametrize("kw", _FLOW_RLS_KW)
def test_flow_matvec_rls_is_self_adjoint(kw):
    # L(w) stays symmetric for a fixed weight map: <Lx, y> == <x, Ly>.
    # Mirrors fastfields-cpu-lib's run_2d_matvec_rls_symmetry.
    rng = np.random.default_rng(52)
    H, W, D = 5, 6, 2
    x = rng.standard_normal((H, W, D))
    y = rng.standard_normal((H, W, D))
    wgt = 0.5 + np.abs(rng.standard_normal((H, W, 1)))
    lx = ff.flow_matvec_rls(x, wgt, ndim=2, **kw)
    ly = ff.flow_matvec_rls(y, wgt, ndim=2, **kw)
    np.testing.assert_allclose(
        float((lx * y).sum()), float((x * ly).sum()), rtol=1e-10
    )
    # ... and the weighting is real: a non-constant w changes the operator.
    assert not np.allclose(lx, ff.flow_matvec(x, ndim=2, **kw))


def test_flow_diag_rls_matches_matvec_rls_on_impulse():
    rng = np.random.default_rng(53)
    H, W, D = 6, 7, 2
    kw = dict(absolute=0.3, membrane=1.0, shears=0.5, div=0.4, ndim=2)
    wgt = 0.5 + np.abs(rng.standard_normal((H, W, 1)))
    d = ff.flow_diag_rls(wgt, **kw)
    assert d.shape == (H, W, D)
    i, j, c = 3, 3, 1
    e = np.zeros((H, W, D))
    e[i, j, c] = 1.0
    o = ff.flow_matvec_rls(e, wgt, **kw)
    np.testing.assert_allclose(d[i, j, c], o[i, j, c], atol=1e-8)


def test_flow_relax_rls_solves_system():
    rng = np.random.default_rng(54)
    H, W, D, hdiag = 6, 7, 2, 8.0
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5, ndim=2)
    hes = np.zeros((H, W, D * (D + 1) // 2))
    hes[..., 0] = hdiag
    hes[..., 1] = hdiag
    grd = rng.standard_normal((H, W, D))
    wgt = 0.5 + np.abs(rng.standard_normal((H, W, 1)))
    x = ff.flow_relax_rls(
        np.zeros((H, W, D)), hes, grd, wgt, nb_iter=250, **kw
    )
    lx = ff.flow_matvec_rls(x, wgt, **kw)
    rel = np.linalg.norm(hdiag * x + lx - grd) / np.linalg.norm(grd)
    assert rel < 3e-3


def test_flow_relax_rls_is_in_place():
    sol = np.zeros((6, 6, 2))
    hes = np.zeros((6, 6, 3))
    hes[..., 0] = hes[..., 1] = 6.0
    grd = np.ones((6, 6, 2))
    wgt = np.ones((6, 6, 1))
    out = ff.flow_relax_rls(
        sol, hes, grd, wgt, membrane=1.0, ndim=2, nb_iter=4
    )
    assert out is sol
    assert np.any(sol != 0.0)


def test_flow_rls_rejects_bending():
    rng = np.random.default_rng(55)
    x = rng.standard_normal((6, 6, 2))
    wgt = np.ones((6, 6, 1))
    with pytest.raises((ValueError, RuntimeError)):
        ff.flow_matvec_rls(x, wgt, bending=1.0, ndim=2)
    with pytest.raises((ValueError, RuntimeError)):
        ff.flow_diag_rls(wgt, bending=1.0, ndim=2)
    with pytest.raises((ValueError, RuntimeError)):
        ff.flow_relax_rls(
            np.zeros((6, 6, 2)),
            np.ones((6, 6, 3)),
            np.ones((6, 6, 2)),
            wgt,
            bending=1.0,
            ndim=2,
        )


# --------------------------------------------------------------------------- #
# Accumulate ops: one in-place kernel, two spellings                          #
#                                                                             #
# The C primitive is in-place only; the out-of-place spelling copies first and#
# runs the same primitive. These tests pin both halves of that contract.      #
# --------------------------------------------------------------------------- #


_ACC_FIELD_KW = dict(absolute=[0.3, 0.4], membrane=[0.7, 0.5], ndim=2)
_ACC_FLOW_KW = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5, ndim=2)


def test_out_of_place_accumulate_does_not_mutate_input():
    rng = np.random.default_rng(11)
    H, W, C = 4, 5, 2
    field = rng.standard_normal((H, W, C))
    base = rng.standard_normal((H, W, C))
    before = base.copy()
    for fn in (ff.field_addmatvec, ff.field_submatvec):
        out = fn(base, field, **_ACC_FIELD_KW)
        np.testing.assert_array_equal(base, before)
        assert out is not base
    for fn in (ff.field_adddiag, ff.field_subdiag):
        out = fn(base, **_ACC_FIELD_KW)
        np.testing.assert_array_equal(base, before)
        assert out is not base


def test_inplace_accumulate_mutates_and_returns_same_array():
    rng = np.random.default_rng(12)
    H, W, C = 4, 5, 2
    field = rng.standard_normal((H, W, C))
    base = rng.standard_normal((H, W, C))
    a = base.copy()
    assert ff.field_addmatvec_(a, field, **_ACC_FIELD_KW) is a
    assert not np.array_equal(a, base)


def test_inplace_and_out_of_place_agree_field():
    rng = np.random.default_rng(13)
    H, W, C = 4, 5, 2
    field = rng.standard_normal((H, W, C))
    base = rng.standard_normal((H, W, C))
    a = base.copy()
    ff.field_addmatvec_(a, field, **_ACC_FIELD_KW)
    b = ff.field_addmatvec(base, field, **_ACC_FIELD_KW)
    np.testing.assert_array_equal(a, b)

    a = base.copy()
    ff.field_subdiag_(a, **_ACC_FIELD_KW)
    b = ff.field_subdiag(base, **_ACC_FIELD_KW)
    np.testing.assert_array_equal(a, b)


def test_inplace_and_out_of_place_agree_flow():
    rng = np.random.default_rng(14)
    H, W = 5, 6
    flow = rng.standard_normal((H, W, 2))
    base = rng.standard_normal((H, W, 2))
    a = base.copy()
    ff.flow_submatvec_(a, flow, **_ACC_FLOW_KW)
    b = ff.flow_submatvec(base, flow, **_ACC_FLOW_KW)
    np.testing.assert_array_equal(a, b)

    a = base.copy()
    ff.flow_adddiag_(a, **_ACC_FLOW_KW)
    b = ff.flow_adddiag(base, **_ACC_FLOW_KW)
    np.testing.assert_array_equal(a, b)


def test_accumulate_matches_reference_composition():
    """The fused primitive must equal the naive `base +/- L(x)` composition."""
    rng = np.random.default_rng(15)
    H, W, C = 5, 6, 2
    field = rng.standard_normal((H, W, C))
    base = rng.standard_normal((H, W, C))
    L = ff.field_matvec(field, **_ACC_FIELD_KW)
    np.testing.assert_allclose(
        ff.field_addmatvec(base, field, **_ACC_FIELD_KW), base + L, atol=1e-12
    )
    np.testing.assert_allclose(
        ff.field_submatvec(base, field, **_ACC_FIELD_KW), base - L, atol=1e-12
    )
    d = ff.field_diag(base.shape, **_ACC_FIELD_KW)
    np.testing.assert_allclose(
        ff.field_adddiag(base, **_ACC_FIELD_KW), base + d, atol=1e-12
    )
    np.testing.assert_allclose(
        ff.field_subdiag(base, **_ACC_FIELD_KW), base - d, atol=1e-12
    )
