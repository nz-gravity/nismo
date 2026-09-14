from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import replace

import numpy as np
import pytest
from tests.helpers import StandardNormalProposal
from tests.integration.test_parallel_replacement import ConstantNormalModel

from nismo import (
    CallableModel,
    NISMOConfig,
    NISMOSampler,
    ParallelSettings,
    SRWalkSettings,
)
from nismo.constrained import BatchEvaluator
from nismo.execution import ExecutionService
from nismo.replacement import ReplacementWorkerContext

pytestmark = pytest.mark.integration

MODES = [
    ParallelSettings(backend="vectorized", queue_size=5, chains_per_task=3),
    ParallelSettings(backend="process", n_workers=2, queue_size=5, chains_per_task=2),
    ParallelSettings(backend="process", scheduler="ordered", n_workers=2, queue_size=5),
    ParallelSettings(
        backend="process",
        scheduler="rolling",
        n_workers=2,
        queue_size=5,
        chains_per_task=2,
        adaptation_interval=3,
    ),
]


def run(parallel, **limits):
    return NISMOSampler(
        model=ConstantNormalModel(),
        importance_morph=StandardNormalProposal(),
        proposal_scheme="s-rwalk",
        n_live=20,
        rng=987,
        tie_policy="randomized_plateau",
        parallel=parallel,
        srwalk_settings=SRWalkSettings(
            n_steps=6, max_steps=24, dynamic_steps=False, profile=True
        ),
    ).run(dlogz=0.00001, max_iterations=19, **limits)


@pytest.mark.parametrize("parallel", MODES)
def test_backends_replay_and_account_queue_tails(parallel):
    first, second = run(parallel), run(parallel)
    np.testing.assert_array_equal(first.dead_points, second.dead_points)
    np.testing.assert_array_equal(first.dead_tie_breakers, second.dead_tie_breakers)
    assert first.rng_state_final == second.rng_state_final
    assert first.queue_diagnostics == second.queue_diagnostics
    assert first.niter == 19
    assert np.all(np.diff(first.dead_tie_breakers) > 0)
    np.testing.assert_array_equal(first.dead_log_x, -np.arange(1, 20) / 20)
    queue = first.queue_diagnostics
    assert queue.queue_jobs_submitted == queue.queue_jobs_completed
    assert queue.queue_candidates_consumed == first.niter
    assert first.n_likelihood_calls == 20 + queue.prefetch_likelihood_calls
    assert queue.prefetch_likelihood_calls == 6 * queue.queue_jobs_completed
    assert queue.wasted_prefetch_likelihood_calls == 6 * (
        queue.queue_jobs_completed - first.niter
    )
    diag = first.execution_diagnostics
    assert 0 < diag.peak_outstanding_chains <= parallel.queue_size
    assert len(diag.task_timings) == queue.queue_jobs_completed
    assert [t.job_id for t in diag.task_timings] == list(
        range(queue.queue_jobs_completed)
    )
    assert all(t.execution_seconds >= 0 for t in diag.task_timings)


@pytest.mark.parametrize("parallel", MODES)
@pytest.mark.parametrize("budget", [25, 26, 39, 51])
def test_all_evaluated_wasted_and_unused_calls_respect_reservations(parallel, budget):
    result = run(parallel, max_likelihood_calls=budget)
    assert result.n_likelihood_calls <= budget
    assert (
        result.n_likelihood_calls
        == 20 + 6 * result.queue_diagnostics.queue_jobs_completed
    )
    assert result.termination_reason == "max_likelihood_calls"
    assert np.all(result.history.mcmc_completed == 6)


class StateDelayedModel(ConstantNormalModel):
    def log_likelihood(self, theta):
        time.sleep(0.0001 * int(np.count_nonzero(theta[:, 0] > 0)))
        return super().log_likelihood(theta)


def test_rolling_retirement_is_independent_of_worker_timing_and_count():
    def sample(workers, delayed):
        return NISMOSampler(
            model=StateDelayedModel() if delayed else ConstantNormalModel(),
            importance_morph=StandardNormalProposal(),
            proposal_scheme="s-rwalk",
            n_live=20,
            rng=22,
            tie_policy="randomized_plateau",
            parallel=ParallelSettings(
                backend="process",
                scheduler="rolling",
                queue_size=4,
                n_workers=workers,
                adaptation_interval=2,
            ),
            srwalk_settings=SRWalkSettings(n_steps=4, max_steps=12),
        ).run(dlogz=0.1, max_iterations=40)

    reference = sample(1, False)
    delayed = sample(2, True)
    np.testing.assert_array_equal(reference.dead_points, delayed.dead_points)
    np.testing.assert_array_equal(
        reference.history.mcmc_completed, delayed.history.mcmc_completed
    )
    assert reference.queue_diagnostics == delayed.queue_diagnostics


@pytest.mark.parametrize("parallel", MODES)
def test_diagnostic_cadence_does_not_change_stopping(parallel):
    dense = run(parallel)
    sparse = run(replace(parallel, diagnostic_interval=7))
    np.testing.assert_array_equal(dense.dead_points, sparse.dead_points)
    np.testing.assert_array_equal(dense.history.logz_total, sparse.history.logz_total)
    assert np.isnan(sparse.history.live_median_log_psi[1])
    assert np.isfinite(sparse.history.live_median_log_psi[-1])


def process_id(theta):
    return float(os.getpid())


def normal_prior(theta):
    return -0.5 * theta[0] ** 2 - 0.5 * np.log(2 * np.pi)


def test_scalar_initialization_uses_the_owned_persistent_pool():
    model = CallableModel(
        ndim=1,
        parameter_names=("x",),
        log_likelihood_fn=process_id,
        log_prior_fn=normal_prior,
        vectorized=False,
    )
    config = NISMOConfig(
        n_live=20,
        proposal_scheme="s-rwalk",
        parallel=ParallelSettings(
            backend="process", n_workers=2, evaluation_chunk_size=2
        ),
    )
    proposal = StandardNormalProposal()
    with ExecutionService() as execution:
        execution.start(ReplacementWorkerContext(config, model, proposal))
        evaluator = BatchEvaluator(model, proposal)
        batch = execution.evaluate(evaluator, np.zeros((20, 1)))
        assert np.all(batch.log_likelihood != os.getpid())
        assert evaluator.n_likelihood_calls == 20
        assert evaluator.n_prior_calls == 20


class FailingModel(ConstantNormalModel):
    def log_likelihood(self, theta):
        if len(theta) < 20:
            raise RuntimeError("deliberate worker error")
        return super().log_likelihood(theta)


@pytest.mark.parametrize("callback_error", [False, True])
def test_owned_pool_closes_for_worker_and_callback_errors(callback_error):
    before = {child.pid for child in mp.active_children()}

    def broken_callback(info):
        raise RuntimeError("deliberate callback error")

    sampler = NISMOSampler(
        model=ConstantNormalModel() if callback_error else FailingModel(),
        importance_morph=StandardNormalProposal(),
        proposal_scheme="s-rwalk",
        n_live=20,
        rng=77,
        tie_policy="randomized_plateau",
        parallel=ParallelSettings(
            backend="process", n_workers=2, scheduler="rolling", queue_size=4
        ),
        srwalk_settings=SRWalkSettings(n_steps=4),
    )
    with pytest.raises(RuntimeError, match="deliberate"):
        sampler.run(
            max_iterations=5, progress=broken_callback if callback_error else False
        )
    assert {child.pid for child in mp.active_children()} <= before


@pytest.mark.parametrize("parallel", MODES[1:])
def test_deadline_drains_and_counts_without_inserting_late_chains(parallel):
    result = run(parallel, max_wall_time=0.05)
    assert result.termination_reason == "max_wall_time"
    assert (
        result.queue_diagnostics.queue_jobs_submitted
        == result.queue_diagnostics.queue_jobs_completed
    )
    assert (
        result.n_likelihood_calls
        == 20 + result.queue_diagnostics.prefetch_likelihood_calls
    )
    assert result.niter == 0


def test_morph_refills_preserve_randomized_stream_and_call_accounting(monkeypatch):
    import nismo.sampler as sampler_module
    from nismo import MORWalkSettings

    # Make the policy comparison deterministic without timing-sensitive assertions.
    original = sampler_module._draw_replacement
    clock = [0.0]

    def monotonic():
        clock[0] += 0.001
        return clock[0]

    monkeypatch.setattr(sampler_module.time, "monotonic", monotonic)

    def expensive_pilot(**kwargs):
        clock[0] += 1000.0
        return original(**kwargs)

    monkeypatch.setattr(sampler_module, "_draw_replacement", expensive_pilot)
    result = NISMOSampler(
        model=ConstantNormalModel(),
        importance_morph=StandardNormalProposal(),
        proposal_scheme="mor-rwalk",
        n_live=20,
        rng=96,
        tie_policy="randomized_plateau",
        mor_rwalk_settings=MORWalkSettings(
            n_proposals=40,
            refill=True,
            refill_min_acceptance=0.01,
            refill_max_batches=1,
        ),
        srwalk_settings=SRWalkSettings(n_steps=4, dynamic_steps=False),
    ).run(dlogz=0.0001, max_iterations=50)
    assert dict(result.execution_diagnostics.phase_seconds)["morph_refill"] > 0
    assert (
        result.n_likelihood_calls
        == 80 + result.queue_diagnostics.prefetch_likelihood_calls
    )
    assert result.n_proposals == 60 + 4 * result.queue_diagnostics.queue_jobs_completed
    assert np.all(np.diff(result.dead_tie_breakers) >= 0)
    assert np.count_nonzero(np.isnan(result.history.mh_acceptance_fraction)) > 0


def test_explicit_worker_threads_are_recorded_without_changing_coordinator():
    from threadpoolctl import threadpool_info

    before = [item["num_threads"] for item in threadpool_info()]
    result = run(ParallelSettings(backend="process", n_workers=2, worker_threads=1))
    assert all(
        all(n == 1 for n in counts)
        for counts in result.execution_diagnostics.worker_thread_counts
    )
    assert [item["num_threads"] for item in threadpool_info()] == before


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "bad"},
        {"scheduler": "bad"},
        {"chains_per_task": 0},
        {"backend": "vectorized", "n_workers": 2},
        {"backend": "compatibility", "scheduler": "rolling"},
        {"worker_threads": 0},
        {"initialization": "process", "n_workers": 1},
    ],
)
def test_execution_settings_reject_incoherent_combinations(options):
    from nismo import ConfigurationError

    with pytest.raises(ConfigurationError):
        ParallelSettings(**options)
