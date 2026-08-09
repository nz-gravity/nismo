# Architecture

NISMO keeps the transformed statistical calculation separate from MorphZ and
from presentation concerns.

```text
user model ───────────────> fixed-q0 evaluator ─┐
fixed importance Morph ──> log_q0 / log_psi0 ──┼─> sampler ─> immutable result
                         └> initial live set    │      │
active proposal Morph ─────> constrained draws ┘      │
periodic live-set refits ───> proposal only            │
                                      quadrature <─────┘

immutable result ──> diagnostics / plotting / user persistence
```

Dependency direction is inward toward small protocols:

- `model.py` defines batched model behavior and callable adaptation.
- `proposals/base.py` defines a normalized proposal contract.
- `proposals/morph.py` is the only module permitted to import or call MorphZ.
- `adaptive.py` schedules live-set proposal refits and records their outcomes.
- `constrained.py` evaluates every candidate against the frozen importance
  Morph while drawing candidates from the independently selected proposal.
- `quadrature.py` contains pure expected-volume evidence arithmetic.
- `stopping.py` contains immutable policies, pure stopping metrics, and
  criterion combination.
- `sampler.py` coordinates iteration and owns no density or plotting details.
- `results.py` validates and freezes complete run outputs.
- `progress.py` adapts optional tqdm or user callbacks to sampler snapshots.
- `diagnostics.py` and `plotting.py` consume results without changing them.

`fixed_morph` uses the importance Morph as the proposal and retains exact
constrained rejection. `adaptive_morph` refits only the proposal object from
the current live set. Because threshold-only adaptive acceptance is not
corrected back to constrained `q0`, it is an explicitly heuristic scheme.
Progress is silent by default; `progress=True` opts into terminal or notebook
output. No library module writes files or displays plots during a run.
