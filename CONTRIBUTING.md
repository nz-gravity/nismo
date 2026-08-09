# Contributing

NISMO is currently a pre-alpha research project.

1. Create a focused branch.
2. Install `.[dev]`.
3. Add deterministic unit tests for numerical and failure behavior.
4. Mark repeated stochastic tests with `statistical` and expensive tests with
   `slow`.
5. Run the commands listed in the README.
6. Explain statistical assumptions and tolerance choices in the change.

Do not silently alter the pseudo-prior, pseudo-likelihood, quadrature, live-set
update, or constrained-draw rule. Phase 3 and later features require explicit
project-owner approval.

