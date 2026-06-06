"""Grid-aware (diagonal-distance) SDR codes — the encoder under test here.

The existing ``rdse`` adapter (``sparse_encoder.py``) encodes the *state index as a
scalar*, so it shares bits between **index-adjacent** states — which on FrozenLake is
*misaligned* with value (state 4 is on-path, state 5 is a hole). That is the
documented FAIL: overlap that ignores the 2D layout.

``CoordinateEncoder`` does the opposite. It encodes the tile's **(row, col)
coordinates**, one sub-encoder per axis, and concatenates them. Two tiles that are
*spatially* close therefore share bits *in both halves*; tiles that are far apart in
either axis share fewer. The resulting code overlap falls off with **grid distance**
(empirically ~ a decreasing function of the diagonal/Chebyshev separation), so the
generalization the TD learner gets is now *spatial* rather than index-based.

Two interchangeable per-axis backends (the user's ask — "both a scalar encoder and a
RDSE encoder"):

  - ``scalar``: the project's real ``ScalarEncoder`` (contiguous active block whose
    position slides with the coordinate). To get overlap between *adjacent* tiles at a
    fixed per-axis width, the active block must span ~``radius_tiles`` tiles, so the
    active-bit count is derived from ``radius_tiles`` (contiguous encoders trade
    sparsity for overlap range).
  - ``rdse``: the project's real ``RandomDistributedScalarEncoder``. Here the axis
    *radius* (``active_bits * resolution``) is independent of the SDR width, so it
    stays genuinely sparse while still overlapping neighbours — set via ``resolution =
    radius_tiles / active_bits``.

Total SDR width is fixed at ``size`` (default **1024**), split evenly across the two
axes (512 bits each).
"""

from __future__ import annotations

import sys
from pathlib import Path


def _add_src_to_path() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


class CoordinateEncoder:
    """state index -> (row, col) -> concat(axis SDRs); grid-distance-aware overlap."""

    def __init__(self, rows: int, cols: int, backend: str = "scalar",
                 size: int = 1024, radius_tiles: float = 2.0,
                 rdse_active_bits: int = 24, seed: int = 0) -> None:
        _add_src_to_path()
        self.rows = rows
        self.cols = cols
        self.backend = backend
        self.size = size
        per_axis = size // 2                 # 512 each for size=1024
        self._row_off = 0                    # row axis occupies [0, per_axis)
        self._col_off = per_axis             # col axis occupies [per_axis, size)
        self._per_axis = per_axis

        if backend == "scalar":
            self._row_enc = self._make_scalar(rows, per_axis, radius_tiles)
            self._col_enc = self._make_scalar(cols, per_axis, radius_tiles)
        elif backend == "rdse":
            self._row_enc = self._make_rdse(per_axis, rdse_active_bits,
                                            radius_tiles, seed)
            self._col_enc = self._make_rdse(per_axis, rdse_active_bits,
                                            radius_tiles, seed + 7)
        else:
            raise ValueError(f"unknown backend: {backend}")

        # active-bit count per state (both axes), for reporting.
        r0 = sum(self._row_enc.encode(0.0))
        c0 = sum(self._col_enc.encode(0.0))
        self.k = r0 + c0
        self.n_bits = size

    # -- backends ---------------------------------------------------------------
    @staticmethod
    def _make_scalar(extent: int, per_axis: int, radius_tiles: float):
        """Contiguous ScalarEncoder over [0, extent-1] at width ``per_axis``.

        active_bits is chosen so the active block spans ~``radius_tiles`` tiles:
        radius ~= active_bits * (range / per_axis)  =>  active_bits ~= radius_tiles *
        per_axis / range. That makes tiles within ``radius_tiles`` of each other share
        bits and tiles farther apart not.
        """
        from encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters

        rng = max(1, extent - 1)
        active = max(1, round(radius_tiles * per_axis / rng))
        active = min(active, per_axis - 1)
        # resolution is a placeholder to satisfy the "exactly one size-arg" check;
        # because size>0 the encoder re-derives resolution from size + active_bits.
        params = ScalarEncoderParameters(
            minimum=0, maximum=rng, active_bits=active, sparsity=0.0,
            size=per_axis, radius=0.0, resolution=1e-9, category=False,
        )
        return ScalarEncoder(params)

    @staticmethod
    def _make_rdse(per_axis: int, active_bits: int, radius_tiles: float, seed: int):
        """RDSE at width ``per_axis``; radius = active_bits*resolution = radius_tiles."""
        from encoder_layer.rdse import RDSEParameters, RandomDistributedScalarEncoder

        resolution = radius_tiles / active_bits
        params = RDSEParameters(size=per_axis, active_bits=active_bits, sparsity=0.0,
                                resolution=resolution, seed=seed + 1)
        return RandomDistributedScalarEncoder(params)

    # -- encode -----------------------------------------------------------------
    def encode(self, state: int) -> tuple[int, ...]:
        row, col = divmod(state, self.cols)
        row_dense = self._row_enc.encode(float(row))
        col_dense = self._col_enc.encode(float(col))
        bits = [self._row_off + i for i, b in enumerate(row_dense) if b]
        bits += [self._col_off + i for i, b in enumerate(col_dense) if b]
        return tuple(sorted(bits))
