"""Streaming preprocessing of a feature matrix ``Phi`` of shape ``(N, D)``.

We never materialize a standardized copy of ``Phi``. Instead, callers
construct a :class:`StandardizedView` once via :func:`fit_view` (which
streams ``Phi`` once to learn the column mean if needed) and then ask the
view for a chunk by row indices. Algorithms downstream iterate via
``view.stream()`` so memory stays proportional to one chunk.

Three modes are supported:

- ``"none"`` -- pass-through. ``mu`` is ``None``.
- ``"center"`` -- subtract the column mean. ``mu`` is the running mean,
  computed in ``float64`` to avoid bias when ``N`` is large.
- ``"center_l2"`` -- centre then L2-normalize each row. This puts every
  sample on the unit sphere in the centred space; downstream the Gram
  matrix is then the cosine-similarity Gram of the centred features.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterator, Literal, Union

import numpy as np
import torch

Mode = Literal["center_l2", "center", "none"]
PhiLike = Union[torch.Tensor, np.ndarray, np.memmap]


@dataclasses.dataclass
class StandardizedView:
    """Lazy, chunked, standardized view of a feature matrix.

    Calling the view with a row-index tensor returns the corresponding rows
    standardized according to ``mode`` and moved to ``device``. ``stream``
    yields ``(row_index_tensor, standardized_chunk)`` pairs covering all
    rows in row order; algorithms walk this iterator instead of holding
    the standardized matrix in memory.

    Attributes:
        Phi: The underlying ``(N, D)`` feature matrix (torch tensor or
            numpy array/memmap). Never mutated.
        mode: One of ``"none"``, ``"center"``, ``"center_l2"``.
        mu: The ``(D,)`` column mean on ``device``, or ``None`` when
            ``mode == "none"``.
        device: Torch device string (``"cuda"`` or ``"cpu"``) where the
            view materializes each chunk.
        chunk_size: Default block size used by :meth:`stream` and by the
            view's internal helpers when iterating ``Phi``.
    """

    Phi: PhiLike
    mode: Mode
    mu: torch.Tensor | None
    device: str
    chunk_size: int

    @property
    def N(self) -> int:
        """Number of rows in ``Phi``."""
        return int(self.Phi.shape[0])

    @property
    def D(self) -> int:
        """Number of columns (feature dimension) in ``Phi``."""
        return int(self.Phi.shape[1])

    def _gather(self, idx: torch.Tensor) -> torch.Tensor:
        """Move rows ``Phi[idx]`` to device as float32 (no standardization)."""
        idx_np = idx.cpu().numpy() if isinstance(idx, torch.Tensor) else np.asarray(idx)
        if isinstance(self.Phi, torch.Tensor):
            chunk = self.Phi[idx_np].to(self.device, dtype=torch.float32, non_blocking=True)
        else:
            arr = np.ascontiguousarray(self.Phi[idx_np], dtype=np.float32)
            chunk = torch.from_numpy(arr).to(self.device, non_blocking=True)
        return chunk

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        """Return the standardized rows ``Phi[idx]`` on ``device``.

        Args:
            idx: Long tensor of row indices into ``Phi``. May live on any
                device; gathering happens on host then is shipped to
                ``self.device``.

        Returns:
            ``torch.Tensor`` of shape ``(len(idx), D)`` and dtype
            ``float32`` on ``self.device``, standardized by ``self.mode``.
        """
        chunk = self._gather(idx)
        if self.mu is not None:
            chunk = chunk - self.mu
        if self.mode == "center_l2":
            norms = chunk.norm(dim=1, keepdim=True).clamp(min=1e-12)
            chunk = chunk / norms
        return chunk

    def stream(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Iterate the full row range in contiguous blocks of ``self.chunk_size``.

        Yields:
            Tuples ``(idx, chunk)`` where ``idx`` is a CPU ``long`` tensor
            of row indices and ``chunk`` is the corresponding standardized
            block on ``self.device``.
        """
        N = self.N
        for start in range(0, N, self.chunk_size):
            stop = min(start + self.chunk_size, N)
            idx = torch.arange(start, stop, dtype=torch.long)
            yield idx, self(idx)


def _streamed_mean(Phi: PhiLike, chunk_size: int, device: str) -> torch.Tensor:
    """Compute the column mean of ``Phi`` in ``float64`` then downcast.

    Single pass over ``Phi`` in ``chunk_size`` blocks. ``float64``
    accumulation matters when ``N`` is large since the per-row contribution
    is small relative to the running sum.

    Args:
        Phi: ``(N, D)`` matrix (torch / numpy).
        chunk_size: Row block size.
        device: Device on which the chunks are summed.

    Returns:
        ``(D,)`` ``float32`` tensor on ``device`` -- the column mean.
    """
    N, D = int(Phi.shape[0]), int(Phi.shape[1])
    acc = torch.zeros(D, dtype=torch.float64, device=device)
    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        if isinstance(Phi, torch.Tensor):
            chunk = Phi[start:stop].to(device, dtype=torch.float32, non_blocking=True)
        else:
            chunk = torch.from_numpy(
                np.ascontiguousarray(Phi[start:stop], dtype=np.float32)
            ).to(device)
        acc += chunk.to(torch.float64).sum(dim=0)
    return (acc / float(N)).to(torch.float32)


def fit_view(
    Phi: PhiLike,
    mode: Mode,
    chunk_size: int,
    device: str,
    out_path: Path | str | None = None,
) -> StandardizedView:
    """Construct a :class:`StandardizedView` over ``Phi``.

    For ``mode in ("center", "center_l2")`` this performs one streamed
    pass over ``Phi`` to compute the column mean (``float64`` accumulator).
    For ``mode == "none"`` there is no pass; the view just slices ``Phi``.

    Args:
        Phi: ``(N, D)`` torch tensor or numpy ndarray / memmap. Float32 is
            recommended; the view casts each chunk to float32 on read.
        mode: One of ``"center_l2"``, ``"center"``, ``"none"``.
        chunk_size: Row block size used throughout the view.
        device: ``"cuda"`` or ``"cpu"``.
        out_path: Optional file path to persist ``{"mu", "mode", "N", "D"}``
            via ``torch.save`` for downstream inspection. Parents are
            created.

    Returns:
        A :class:`StandardizedView` referencing ``Phi`` (no copy).

    Raises:
        ValueError: If ``mode`` is not one of the three allowed values.
    """
    if mode not in ("center_l2", "center", "none"):
        raise ValueError(f"unknown mode: {mode}")
    N, D = int(Phi.shape[0]), int(Phi.shape[1])
    mu: torch.Tensor | None = None
    if mode in ("center", "center_l2"):
        mu = _streamed_mean(Phi, chunk_size=chunk_size, device=device)
    view = StandardizedView(Phi=Phi, mode=mode, mu=mu, device=device, chunk_size=chunk_size)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"mu": (mu.cpu() if mu is not None else None), "mode": mode, "N": N, "D": D},
            out_path,
        )
    return view
