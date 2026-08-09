# Grouped Morph Gaussian-shell regression

The small shell regression uses:

- normalized two-dimensional standard-Gaussian prior;
- likelihood
  \(\log L=-\frac12[(\|\theta\|-2)/0.2]^2\);
- analytic evidence calculated by one-dimensional radial quadrature;
- 400 fixed-seed approximate posterior training samples;
- one MorphZ group containing both `x` and `y`;
- seed `72`, `n_live=25`, remaining-log-evidence tolerance `dlogz=0.1`, batch
  size `16`;
- at most 400 iterations and 10,000 proposals per replacement.

The test asserts finite Morph densities, monotone thresholds, scientific
termination, and log evidence within `0.3` of radial quadrature. It is marked
`slow` to keep the marker policy explicit, although it is small.

Run:

```bash
MPLCONFIGDIR=/tmp/nismo-mpl PYTHONPATH=src python -m pytest \
  tests/statistical/test_benchmarks.py::test_grouped_morph_gaussian_shell_regression \
  -m slow
```

No owner-supplied shell group JSON or stored reference output was present.
Accordingly, this fixture has no hard-coded external path and does not claim to
reproduce an unavailable prototype configuration.
