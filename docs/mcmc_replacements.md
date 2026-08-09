# Constrained MCMC replacements

For standalone equation-by-equation audits, see the current standard
[`rwalk` implementation](rwalk_implementation.tex) and the separate
[`s-rwalk` implementation](s_rwalk_implementation.tex).

NISMO always defines its pseudo-likelihood with one fixed importance density:

\[
\log\Psi_0(\theta)
=\log L(\theta)+\log\pi(\theta)-\log q_0(\theta).
\]

At discarded threshold \(\lambda\), the statistically correct replacement
target is

\[
p_\lambda(\theta)\propto
q_0(\theta)\mathbf 1[\log\Psi_0(\theta)>\lambda].
\]

This is a constrained prior only in the special case \(q_0=\pi\). The
`"rwalk"`, `"s-rwalk"`, and `"en-rwalk"` proposal schemes are
Metropolis-Hastings kernels invariant under this constrained fixed-\(q_0\)
density.

## Constraint and acceptance

The sampler first identifies the worst point using stored `live_log_psi0`
ordering. That point defines the threshold. It is not an MCMC start: under
strict ordering it has `log_psi0 == threshold` and lies outside the support.
Standard random walk starts from one uniformly selected eligible survivor.
The statistically specified random walk does the same.
Ensemble random walk starts from distinct eligible survivors selected uniformly
without replacement.

Every proposed parameter point is evaluated once against the model and the
original `importance_morph`. If it passes the pseudo-likelihood constraint,
the generic log Metropolis--Hastings acceptance ratio is

```python
log_alpha = min(
    0.0,
    proposed.log_q0 - current.log_q0 + log_hastings_ratio,
)
accept = np.log(rng.random()) < log_alpha
```

The Hastings term is zero for symmetric proposals. It is nonzero for the
ensemble stretch move described below.

Likelihood, prior, and pseudo-likelihood values affect the constraint. They are
not the MH density ratio. In particular, passing the constraint does not imply
acceptance.

For `tie_policy="randomized_plateau"`, the state is
\((\theta,t)\), with \(t\sim U(0,1)\). Every proposal draws a new tie breaker.
Equal-pseudo-likelihood candidates pass only if their proposed tie exceeds the
discarded threshold tie. Acceptance updates both fields; rejection retains
both.

## Proposal geometry

Standard `rwalk` constructs Dynesty's single bounding ellipsoid from the
complete live set, including the point that defines the current threshold.
Dynesty's internal eigenvalue and condition-number repair produces finite axes
for rank-deficient live sets. The axes are cached and rebuilt after roughly
`walks * n_live` random-walk calls.

`s-rwalk` and the DE/Gaussian `en-rwalk` moves use a regularized covariance
built from frozen survivors:

\[
C_{\rm reg}
=(1-\rho)C+\rho\,\operatorname{diag}(C)+\epsilon I.
\]

The discarded point defines the threshold but is excluded from ensemble
initialization and covariance estimation. Geometry remains fixed during each
complete replacement and is built lazily for `en-rwalk`; a stretch-only
replacement never computes it. Parameters are proposed in their full physical
space. NISMO does not clip to a prior boundary or redraw until a point enters the
prior, because either operation would change the proposal kernel. A proposal
with zero prior or likelihood is rejected through the ordinary constraint.

## Standard random walk

```python
from nismo import NISMOSampler, RWalkSettings

sampler = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="rwalk",
    rwalk_settings=RWalkSettings(
        walks=50,
        facc=0.5,
    ),
    n_live=200,
    rng=42,
)
```

For dimension \(D\), each transition proposes

\[
\theta'=\theta+sAr,\qquad r\sim\operatorname{Uniform}(B_D),
\]

where \(A\) contains the bounding-ellipsoid axes and \(B_D\) is the unit
\(D\)-ball. Scale starts at 1 and, after each completed replacement, follows
Dynesty's update

\[
s_{k+1}=s_k\exp\!\left(\frac{f_k-f_0}{D f_0}\right),
\]

where \(f_0\) is `facc`. The configured value is clamped to
`[1 / walks, 1]`. Omitting `walks` uses `D + 20`; explicit values have a
minimum of two. NISMO attempts exactly `walks` transitions and returns the
actual final state. If every proposal is rejected, that final state is the
valid starting survivor; movement is never forced.

Dynesty can refresh dimensions beyond `ncdim` from a unit-cube prior. NISMO
supports arbitrary correlated importance Morphs without such a transform, so
`ncdim` must be omitted or equal to the complete model dimension.

## Statistically specified Gaussian random walk

```python
from nismo import NISMOSampler, SRWalkSettings

sampler = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="s-rwalk",
    srwalk_settings=SRWalkSettings(
        n_steps=50,
        facc=0.5,
    ),
    n_live=200,
    rng=42,
)
```

For dimension \(D\), `s-rwalk` freezes the regularized survivor covariance
factor \(LL^\mathsf{T}=C_{\rm reg}\) for a complete replacement and proposes

\[
\theta'=\theta+sLz,\qquad z\sim\mathcal N(0,I).
\]

The initial scale defaults to \(2.38/\sqrt D\), or to an explicitly configured
positive `scale`. After each complete chain it uses the same acceptance-target
recursion as `rwalk`; `facc` defaults to 0.5 and is clamped to
`[1 / n_steps, 1]`. Exactly `n_steps` transitions are attempted. Covariance
shrinkage and jitter are configurable, and rank-deficient or
lower-sample-than-dimension live sets are handled by deterministic eigenvalue
flooring.

## Ensemble move mixture

```python
from nismo import EnsembleMoveWeights, EnsembleRWalkSettings, NISMOSampler

sampler = NISMOSampler(
    model=model,
    importance_morph=importance_morph,
    proposal_scheme="en-rwalk",
    ensemble_rwalk_settings=EnsembleRWalkSettings(
        n_walkers=16,
        n_sweeps=6,
        move_weights=EnsembleMoveWeights(
            de=0.60,
            stretch=0.25,
            gaussian=0.15,
        ),
    ),
    n_live=200,
    rng=42,
)
```

`n_walkers` must be even, at least four, and no greater than `n_live - 1`.
Weights are non-negative relative weights, do not need to sum to one, and are
normalized internally after zero-weight moves are omitted. At least one must be
positive. Move selection happens once per half-update and is fixed and
state-independent; acceptance statistics do not adapt the weights.

The default is the general 60/25/15 mixture:

```python
settings = EnsembleRWalkSettings()
```

The former DE-only behavior remains selectable explicitly and does not consume
a move-selection RNG draw:

```python
settings = EnsembleRWalkSettings(
    move_weights=EnsembleMoveWeights(
        de=1.0,
        stretch=0.0,
        gaussian=0.0,
    ),
)
```

A DE-dominant option for a higher-dimensional problem can be expressed without
claiming it is universally optimal:

```python
settings = EnsembleRWalkSettings(
    n_walkers=20,
    n_sweeps=6,
    move_weights=EnsembleMoveWeights(
        de=0.75,
        stretch=0.10,
        gaussian=0.15,
    ),
)
```

Pure stretch is also valid:

```python
settings = EnsembleRWalkSettings(
    move_weights=EnsembleMoveWeights(
        de=0.0,
        stretch=1.0,
        gaussian=0.0,
    ),
)
```

Each sweep randomly permutes and splits the ensemble into equal halves. The
first half is updated against a frozen second half; the second is then updated
against the now-frozen first half. The selected move proposes every active
walker in one vectorized batch.

### Differential evolution

Walkers use two distinct, ordered references from the complementary half:

\[
\theta_i'=\theta_i+\gamma(\theta_j-\theta_k)
          +\sigma_\epsilon Lz.
\]

The default is \(\gamma=2.38/\sqrt{2D}\). Nonzero symmetric covariance-shaped
jitter is always present, so this move has zero Hastings correction.

### Stretch

One complementary walker $j$ is selected for each active walker and

\[
z=\frac{((a-1)u+1)^2}{a},\quad u\sim U(0,1),\qquad
\theta_i'=\theta_j+z(\theta_i-\theta_j),
\]

where `stretch_scale` is $a>1$, defaulting to 2. This is not symmetric. Its
mandatory Hastings correction is

\[
\log\frac{Q(\theta_i\mid\theta_i')}{Q(\theta_i'\mid\theta_i)}
=(D-1)\log z.
\]

Omitting the corresponding $z^{D-1}$ factor would target the wrong
distribution.

### Gaussian covariance

The local Gaussian move is

\[
\theta_i'=\theta_i+sLz_i,\qquad z_i\sim\mathcal N(0,I).
\]

Its scale defaults to $s=2.38/\sqrt D$, or uses a positive explicit
`gaussian_scale`. It is symmetric and has zero Hastings correction. The same
regularized survivor factor $L$ used by DE jitter is computed at most once and
frozen for all sweeps in a replacement; it is never recomputed after accepted
ensemble moves.

After the configured sweeps, one walker is selected uniformly from the complete
final ensemble. Unchanged and rejected walkers remain eligible for output.

## Limits and diagnostics

One random-walk transition and one ensemble walker candidate each count as one
proposal. A successful replacement therefore costs:

- `rwalk`: `walks` (default `ndim + 20`);
- `s-rwalk`: `n_steps` (default 25);
- `en-rwalk`: `n_walkers * n_sweeps`.

NISMO checks obvious proposal and likelihood-call incompatibilities before
starting. It checks the deadline before every scalar transition or ensemble
half-batch. An interrupted evolution fails the replacement and leaves the
worst live point untouched; it never returns a shortened chain.

`RunHistory` separates nested replacement efficiency from MCMC behavior:

- `acceptance_fraction` retains cumulative completed-replacements/proposals;
- `constraint_pass_fraction` is constraint passes divided by all candidates;
- `mh_acceptance_fraction` is MH acceptances divided by all candidates;
- `mcmc_accepted` counts accepted transitions;
- `mcmc_moved` is one for a single chain that moved, or the number of
  ensemble walkers that moved at least once;
- `mcmc_completed` counts transitions for `"rwalk"` and `"s-rwalk"`, and
  complete sweeps for `"en-rwalk"`.

The two fractions are `NaN` and counts are zero for non-MCMC schemes.

For `en-rwalk`, `result.ensemble_move_history` adds immutable per-move matrices:

```python
move_history = result.ensemble_move_history
assert move_history.names == ("de", "stretch", "gaussian")
proposed = move_history.proposed  # shape (result.niter, 3), read-only
valid = move_history.valid
accepted = move_history.accepted
moved = move_history.moved
```

Each row's proposed, valid, and accepted counts sum to the aggregate iteration
counts. A move's `moved` value counts distinct walkers it moved during that
replacement. A walker can appear in more than one move column if it accepted
different move types.

## Mixing limitations

A finite-length Metropolis kernel is invariant under the constrained target,
but its replacement is generally correlated with existing live points. It is
not an independent constrained draw, and invariance alone does not establish
adequate mixing. Calibrate `walks`, `n_steps`, `n_sweeps`, walker count, and
`n_live` with repeated complete runs and compare evidence, posterior summaries,
and cost across seeds.

The `rwalk` implementation is adapted from Dynesty 3.1.0. Its runtime
`citations` entry remains Skilling (2006); source attribution and Dynesty's MIT
license are distributed with NISMO.

Low MH acceptance, low constraint-pass fractions, or many unchanged walkers
can indicate poor mixing. Do not repair those symptoms by returning only moved
walkers, requiring at least one acceptance, or selecting a high-likelihood
state; all such conditioning changes the target.
