"""Typed exceptions raised by NISMO."""


class NISMOError(Exception):
    """Base class for package-specific errors."""


class ConfigurationError(NISMOError, ValueError):
    """Raised when sampler configuration is inconsistent or out of range."""


class InvalidModelOutput(NISMOError, ValueError):
    """Raised when a model returns invalid values or array shapes."""


class InvalidProposalOutput(NISMOError, ValueError):
    """Raised when a proposal returns invalid values or array shapes."""


class ProposalSupportError(NISMOError):
    """Raised when Morph has no density where the target integrand is finite."""


class MissingOptionalDependency(NISMOError, ImportError):
    """Raised when a requested optional integration is unavailable."""


class NumericalInvariantError(NISMOError, ArithmeticError):
    """Raised when evidence arithmetic violates a required invariant."""
