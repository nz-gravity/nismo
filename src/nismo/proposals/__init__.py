"""Proposal interfaces and adapters."""

from .base import Proposal, RefittableProposal
from .morph import MorphMetadata, MorphProposal

__all__ = ["MorphMetadata", "MorphProposal", "Proposal", "RefittableProposal"]
