"""Periodic proposal-Morph refitting with a fixed importance Morph."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .proposals import MorphMetadata, Proposal, RefittableProposal
from .results import ProposalUpdateRecord


def _metadata(proposal: Proposal) -> MorphMetadata | None:
    metadata = getattr(proposal, "metadata", None)
    return metadata if isinstance(metadata, MorphMetadata) else None


class AdaptiveMorphController:
    """Own the replaceable proposal Morph and its scheduled update records."""

    def __init__(
        self,
        *,
        importance_morph: RefittableProposal,
        update_interval: int,
    ) -> None:
        if update_interval < 1:
            raise ValueError("update_interval must be positive")
        self.importance_morph = importance_morph
        self.proposal_morph: Proposal = importance_morph
        self.update_interval = update_interval
        self.revision = 0
        self.update_attempts = 0
        self.update_failures = 0
        self._next_update = update_interval
        self._records: list[ProposalUpdateRecord] = []

    @property
    def records(self) -> tuple[ProposalUpdateRecord, ...]:
        """Return immutable update outcomes in scheduled order."""
        return tuple(self._records)

    def update_if_due(
        self,
        *,
        iteration: int,
        live_theta: NDArray[np.float64],
    ) -> Proposal:
        """Refit at the scheduled boundary, retaining the old proposal on failure."""
        if iteration != self._next_update:
            return self.proposal_morph

        training = np.array(live_theta, dtype=float, copy=True)
        self.update_attempts += 1
        try:
            candidate = self.importance_morph.refit(training)
            if not isinstance(candidate, Proposal):
                raise TypeError("refit must return a normalized Proposal")
            if candidate.ndim != self.importance_morph.ndim:
                raise ValueError(
                    "refitted proposal dimension does not match importance Morph"
                )
        except Exception as error:
            self.update_failures += 1
            self._records.append(
                ProposalUpdateRecord(
                    iteration=iteration,
                    success=False,
                    active_revision=self.revision,
                    n_training=len(training),
                    proposal_metadata=_metadata(self.proposal_morph),
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
        else:
            self.proposal_morph = candidate
            self.revision += 1
            self._records.append(
                ProposalUpdateRecord(
                    iteration=iteration,
                    success=True,
                    active_revision=self.revision,
                    n_training=len(training),
                    proposal_metadata=_metadata(candidate),
                )
            )
        self._next_update += self.update_interval
        return self.proposal_morph
