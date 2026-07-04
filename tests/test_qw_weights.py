from __future__ import annotations

from basecall_deletion.training.weights import qw_default


def test_qw_thresholds() -> None:
    assert qw_default(0.9700) == 1.0
    assert qw_default(0.9699) == 0.8
    assert qw_default(0.9500) == 0.8
    assert qw_default(0.9499) == 0.5
    assert qw_default(0.9000) == 0.5
    assert qw_default(0.8999) == 0.2

