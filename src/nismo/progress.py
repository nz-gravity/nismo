"""Optional terminal progress reporting for NISMO runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from .exceptions import MissingOptionalDependency

ProgressValue = float | int
ProgressInfo = Mapping[str, ProgressValue]
ProgressCallback = Callable[[ProgressInfo], None]
ProgressOption = bool | ProgressCallback | None


class ProgressReporter(Protocol):
    """Internal interface shared by terminal and callback reporters."""

    @property
    def is_active(self) -> bool:
        """Whether the reporter consumes per-iteration update payloads."""

    def update(self, info: ProgressInfo) -> None:
        """Consume one completed-iteration progress snapshot."""

    def close(self, termination_reason: str) -> None:
        """Close resources and display the terminal reason."""


class _NullProgress:
    @property
    def is_active(self) -> bool:
        return False

    def update(self, info: ProgressInfo) -> None:
        del info

    def close(self, termination_reason: str) -> None:
        del termination_reason


class _CallbackProgress:
    def __init__(self, callback: ProgressCallback) -> None:
        self._callback = callback

    @property
    def is_active(self) -> bool:
        return True

    def update(self, info: ProgressInfo) -> None:
        self._callback(info)

    def close(self, termination_reason: str) -> None:
        del termination_reason


class _TqdmProgress:
    def __init__(self, *, n_live: int) -> None:
        try:
            from tqdm.auto import tqdm
        except ImportError as error:  # pragma: no cover - environment dependent
            raise MissingOptionalDependency(
                "progress=True requires tqdm; install NISMO with the 'progress' extra"
            ) from error
        self._bar = tqdm(
            total=None,
            desc=f"NISMO nlive={n_live}",
            unit="it",
            dynamic_ncols=True,
            mininterval=0.2,
            delay=0.2,
            bar_format="{desc}: iter: {n} [{elapsed}, {rate_fmt}{postfix}]",
        )
        self._postfix: dict[str, str | int] = {}

    @property
    def is_active(self) -> bool:
        return True

    def update(self, info: ProgressInfo) -> None:
        iteration = int(info["iteration"])
        self._postfix = {
            "ncall": int(info["likelihood_calls"]),
            "eff": f"{info['efficiency_percent']:.1f}%",
            "logZ": f"{info['logz']:.3f}",
            "logZerr": f"{info['logzerr']:.3f}",
            "H": f"{info['information']:.3f}",
        }
        if "criterion_remaining_dlogz_met" in info:
            self._postfix["dlog(z)"] = f"{info['remaining_dlogz']:.2e}"
        if "criterion_live_logz_error_met" in info:
            self._postfix["liveErr"] = f"{info['live_logz_error']:.2e}"
        if "criterion_live_ess_met" in info:
            self._postfix["ESSlive"] = f"{info['live_ess']:.1f}"
        if int(info["proposal_update_failures"]):
            self._postfix["propfail"] = int(info["proposal_update_failures"])
        if int(info["proposal_revision"]):
            self._postfix["prop"] = int(info["proposal_revision"])
        if "criterion_logz_stability_met" in info:
            self._postfix["stable"] = f"{info['logz_stability']:.2e}"
        self._postfix["logPsi*"] = f"{info['threshold']:.3f}"
        self._bar.set_postfix(self._postfix, refresh=False)
        self._bar.update(max(0, iteration - self._bar.n))

    def close(self, termination_reason: str) -> None:
        del termination_reason
        self._bar.delay = 0.0
        self._bar.close()


def create_progress_reporter(
    progress: ProgressOption,
    *,
    max_iterations: int,
    n_live: int,
) -> ProgressReporter:
    """Create a silent, callback, or optional tqdm progress reporter.

    Parameters
    ----------
    progress
        ``False``/``None`` for silence, ``True`` for the standard terminal live
        display, or a callable receiving progress mappings.
    max_iterations
        Retained for reporter construction compatibility. Hard iteration limits
        are not represented as a convergence percentage.
    n_live
        Static live-point count displayed in the bar description.
    """
    if progress is True:
        return _TqdmProgress(
            n_live=n_live,
        )
    if progress is False or progress is None:
        return _NullProgress()
    if callable(progress):
        return _CallbackProgress(progress)
    raise TypeError("progress must be a bool, callable, or None")
