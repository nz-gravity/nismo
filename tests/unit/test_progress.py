from __future__ import annotations

import sys

import pytest

from nismo import MissingOptionalDependency
from nismo.progress import create_progress_reporter

pytestmark = pytest.mark.unit


def test_progress_true_reports_missing_optional_tqdm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    with pytest.raises(MissingOptionalDependency, match=r"progress.*extra"):
        create_progress_reporter(
            True,
            max_iterations=10,
            n_live=5,
        )


def test_silent_progress_reporter_accepts_snapshots() -> None:
    reporter = create_progress_reporter(
        False,
        max_iterations=10,
        n_live=5,
    )
    reporter.update({"iteration": 1})
    reporter.close("remaining_evidence")
