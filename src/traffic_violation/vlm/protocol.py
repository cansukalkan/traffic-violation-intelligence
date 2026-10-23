"""Model-neutral interface for visual candidate review."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from traffic_violation.events import ViolationCandidate


class Describer(Protocol):
    """Describe one visual candidate; implementations may be local or remote."""

    def describe(
        self,
        frame_bgr: NDArray[np.uint8],
        candidate: ViolationCandidate,
    ) -> str: ...
