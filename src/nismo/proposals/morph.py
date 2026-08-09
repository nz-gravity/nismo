"""MorphZ ``GroupKDE`` adapter implementing the normalized proposal contract."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..exceptions import InvalidProposalOutput, MissingOptionalDependency


@dataclass(frozen=True, slots=True)
class MorphMetadata:
    """Immutable description of a fitted MorphZ proposal."""

    n_training: int
    ndim: int
    parameter_names: tuple[str, ...]
    kde_bw: str
    min_tc: float | None
    top_k_greedy: int
    group_source: str
    selected_groups: tuple[tuple[str, ...], ...]
    single_parameters: tuple[str, ...]
    morphz_version: str
    rng_note: str


class MorphProposal:
    """Adapt a fixed normalized ``morphZ.GroupKDE`` to NISMO.

    Instances must be constructed with :meth:`fit`.

    Notes
    -----
    MorphZ 0.4.1 accepts an integer seed in ``resample``. NISMO derives that
    integer from the run's explicit :class:`numpy.random.Generator`. The
    inspected MorphZ implementation temporarily seeds NumPy's legacy global
    RNG and restores its previous state before returning. NISMO never directly
    reseeds global state.
    """

    def __init__(
        self,
        backend: Any,
        metadata: MorphMetadata,
        refit_kwargs: dict[str, Any] | None = None,
        logpdf_batch_mode: Literal["native", "transposed", "scalar"] = "scalar",
    ) -> None:
        self._backend = backend
        self.metadata = metadata
        self.ndim = metadata.ndim
        self._refit_kwargs = deepcopy(refit_kwargs)
        self._logpdf_batch_mode = logpdf_batch_mode

    @staticmethod
    def _resolve_logpdf_batch_mode(
        backend: Any,
        samples: NDArray[np.float64],
    ) -> Literal["native", "transposed", "scalar"]:
        """Find a verified batch convention without trusting backend docs.

        MorphZ 0.4.1 accepts batches only after a transpose despite documenting
        the opposite orientation.  Probe both conventions against scalar
        values at fit time so a future backend can use its native convention
        and an unknown backend safely falls back to row-wise evaluation.
        """
        n_probe = 2 if samples.shape[1] != 2 else 3
        probe = np.array(
            [samples[index % len(samples)] for index in range(n_probe)],
            dtype=float,
        )
        expected = np.asarray(
            [backend.logpdf(point) for point in probe],
            dtype=float,
        )
        batch_modes: tuple[
            tuple[Literal["native", "transposed"], NDArray[np.float64]],
            ...,
        ] = (
            ("native", probe),
            ("transposed", probe.T),
        )
        for mode, value in batch_modes:
            try:
                result = np.asarray(backend.logpdf(value), dtype=float)
            except (TypeError, ValueError):
                continue
            if result.shape == expected.shape and np.allclose(
                result,
                expected,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                return mode
        return "scalar"

    @classmethod
    def fit(
        cls,
        posterior_samples: ArrayLike,
        *,
        morph_type: str | None = None,
        group_file: str | Path | None = None,
        groups: Sequence[Sequence[object]] | None = None,
        param_names: Sequence[str] | None = None,
        kde_bw: str | float | dict[str, float] = "silverman",
        min_tc: float | None = None,
        top_k_greedy: int = 1,
    ) -> MorphProposal:
        """Fit one fixed MorphZ ``GroupKDE``.

        Parameters
        ----------
        posterior_samples
            Finite training values with shape ``(n_samples, ndim)``. The input
            is copied and is never used as the initial live set.
        morph_type
            Automatic MorphZ grouped approximation in ``"{k}_group"`` form,
            such as ``"2_group"`` or ``"3_group"``. MorphZ computes k-order
            total correlations and ``GroupKDE`` performs greedy disjoint-group
            selection. Temporary TC artifacts are removed after fitting.
        group_file
            JSON grouping definition accepted by MorphZ. NISMO reads the file
            and passes its contents in memory to prevent MorphZ from writing a
            sibling selection file.
        groups
            In-memory MorphZ grouping definition. Specify at most one of
            ``group_file`` and ``groups``. An empty sequence selects independent
            one-dimensional KDEs.
        param_names
            Names corresponding to sample columns.
        kde_bw, min_tc, top_k_greedy
            Values passed unchanged to ``morphZ.GroupKDE``.

        Returns
        -------
        MorphProposal
            A proposal whose sampling and log density use the same fitted
            normalized GroupKDE.

        Raises
        ------
        MissingOptionalDependency
            If MorphZ is unavailable.
        ValueError
            If training values or grouping inputs are invalid.
        """
        samples = np.array(posterior_samples, dtype=float, copy=True)
        if samples.ndim != 2:
            raise ValueError("posterior_samples must have shape (n_samples, ndim)")
        if samples.shape[0] < 2 or samples.shape[1] < 1:
            raise ValueError(
                "posterior_samples must contain at least two rows and one column"
            )
        if not np.all(np.isfinite(samples)):
            raise ValueError("posterior_samples must contain only finite values")
        grouping_inputs = sum(
            option is not None for option in (morph_type, group_file, groups)
        )
        if grouping_inputs > 1:
            raise ValueError("specify only one of morph_type, group_file, and groups")
        if param_names is None:
            names = tuple(f"param_{index}" for index in range(samples.shape[1]))
        else:
            names = tuple(str(name) for name in param_names)
        if len(names) != samples.shape[1]:
            raise ValueError("param_names length must match posterior sample columns")
        if len(set(names)) != len(names):
            raise ValueError("param_names must be unique")
        if isinstance(top_k_greedy, bool) or top_k_greedy < 1:
            raise ValueError("top_k_greedy must be a positive integer")

        try:
            import morphZ
        except ImportError as error:  # pragma: no cover - environment dependent
            raise MissingOptionalDependency(
                "MorphProposal requires MorphZ; install NISMO with the 'morph' extra"
            ) from error

        if morph_type is not None:
            match = re.fullmatch(r"([1-9][0-9]*)_group", morph_type)
            if match is None:
                raise ValueError(
                    "morph_type must use MorphZ's '<k>_group' form, "
                    "for example '2_group'; literal 'n_group' is not valid"
                )
            n_order = int(match.group(1))
            if n_order < 2:
                raise ValueError("group order in morph_type must be at least 2")
            if n_order > samples.shape[1]:
                raise ValueError(
                    "group order in morph_type cannot exceed posterior dimension"
                )
            with TemporaryDirectory(prefix="nismo-morphz-tc-") as temporary_path:
                morphZ.Nth_TC.compute_and_save_tc(
                    samples,
                    names=list(names),
                    n_order=n_order,
                    out_path=temporary_path,
                )
                tc_path = Path(temporary_path) / f"params_{n_order}-order_TC.json"
                if not tc_path.is_file():
                    raise RuntimeError(
                        "MorphZ total-correlation computation did not create "
                        f"{tc_path.name}"
                    )
                with tc_path.open(encoding="utf-8") as stream:
                    group_definition = json.load(stream)
            group_source = f"automatic:{morph_type}"
        elif group_file is not None:
            group_path = Path(group_file)
            with group_path.open(encoding="utf-8") as stream:
                group_definition = json.load(stream)
            group_source = str(group_path)
        elif groups is not None:
            group_definition = list(groups)
            group_source = "in-memory"
        else:
            group_definition = []
            group_source = "independent-default"

        backend = morphZ.GroupKDE(
            samples,
            param_tc=group_definition,
            param_names=list(names),
            kde_bw=kde_bw,
            min_tc=min_tc,
            verbose=False,
            top_k_greedy=top_k_greedy,
        )
        metadata = MorphMetadata(
            n_training=samples.shape[0],
            ndim=samples.shape[1],
            parameter_names=names,
            kde_bw=repr(kde_bw),
            min_tc=min_tc,
            top_k_greedy=top_k_greedy,
            group_source=group_source,
            selected_groups=tuple(
                tuple(str(name) for name in group.get("names", ()))
                for group in getattr(backend, "groups", ())
            ),
            single_parameters=tuple(
                str(name) for name in getattr(backend, "singles", ())
            ),
            morphz_version=str(getattr(morphZ, "__version__", "unknown")),
            rng_note=(
                "NISMO derives an integer seed from its Generator for each MorphZ "
                "resample call; MorphZ 0.4.1 restores legacy global RNG state."
            ),
        )
        if morph_type is not None:
            refit_grouping: dict[str, Any] = {"morph_type": morph_type}
        else:
            # Retain the already loaded in-memory definition. Refit operations
            # must not depend on a group file continuing to exist.
            refit_grouping = {"groups": deepcopy(group_definition)}
        refit_kwargs = {
            **refit_grouping,
            "param_names": names,
            "kde_bw": deepcopy(kde_bw),
            "min_tc": min_tc,
            "top_k_greedy": top_k_greedy,
        }
        return cls(
            backend,
            metadata,
            refit_kwargs,
            cls._resolve_logpdf_batch_mode(backend, samples),
        )

    def refit(
        self,
        training_theta: NDArray[np.float64],
    ) -> MorphProposal:
        """Fit a new Morph with the original settings without mutating this one."""
        if self._refit_kwargs is None:
            raise RuntimeError("this MorphProposal does not retain refit settings")
        return type(self).fit(training_theta, **deepcopy(self._refit_kwargs))

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw independent points from the fitted MorphZ density.

        Parameters
        ----------
        n
            Positive number of points.
        rng
            Run-owned random generator used to derive MorphZ's integer seed.

        Returns
        -------
        numpy.ndarray
            Finite points with shape ``(n, ndim)``.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        points = np.asarray(
            self._backend.resample(n, random_state=seed),
            dtype=float,
        )
        if points.shape != (n, self.ndim):
            raise InvalidProposalOutput(
                f"MorphZ resample returned {points.shape}, expected {(n, self.ndim)}"
            )
        if not np.all(np.isfinite(points)):
            raise InvalidProposalOutput("MorphZ resample returned NaN or infinity")
        return points

    def log_prob(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate the normalized fitted MorphZ log density.

        The adapter probes the backend's batch orientation when fitting and
        uses it only after verifying agreement with scalar values.  Unknown
        backends retain the conservative row-wise fallback.
        """
        points = np.asarray(theta, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != self.ndim:
            raise InvalidProposalOutput(
                f"theta must have shape (n, {self.ndim}), got {points.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise InvalidProposalOutput("theta contains NaN or infinity")
        if self._logpdf_batch_mode == "native":
            values = np.asarray(self._backend.logpdf(points), dtype=float)
        elif self._logpdf_batch_mode == "transposed":
            values = np.asarray(self._backend.logpdf(points.T), dtype=float)
        else:
            values = np.asarray(
                [self._backend.logpdf(point) for point in points],
                dtype=float,
            )
        if values.shape != (len(points),):
            raise InvalidProposalOutput(
                f"MorphZ logpdf returned {values.shape}, expected {(len(points),)}"
            )
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise InvalidProposalOutput("MorphZ logpdf returned NaN or +infinity")
        return values
