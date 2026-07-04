from __future__ import annotations


def qw_default(identity: float) -> float:
    """Map offline teacher-to-reference confidence to the default QW-CTC weight."""
    q = float(identity)
    if q >= 0.97:
        return 1.0
    if q >= 0.95:
        return 0.8
    if q >= 0.90:
        return 0.5
    return 0.2

