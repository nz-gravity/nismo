from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nismo import MorphProposal
from nismo.exceptions import InvalidProposalOutput

pytestmark = pytest.mark.unit


def test_morph_adapter_sampling_density_and_seed_reproducibility() -> None:
    training = np.random.default_rng(4).normal(size=(250, 1))
    proposal = MorphProposal.fit(
        training,
        groups=[],
        param_names=("x",),
        kde_bw="silverman",
    )
    first = proposal.sample(20, np.random.default_rng(91))
    second = proposal.sample(20, np.random.default_rng(91))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (20, 1)
    log_q = proposal.log_prob(first)
    assert log_q.shape == (20,)
    assert np.all(np.isfinite(log_q))
    assert proposal.metadata.n_training == 250
    assert proposal.metadata.morphz_version != "unknown"


def test_morph_adapter_copies_and_validates_training_data() -> None:
    training = np.random.default_rng(5).normal(size=(100, 1))
    proposal = MorphProposal.fit(training, groups=[])
    training[:] = np.nan
    assert np.isfinite(proposal.log_prob(np.array([[0.0]]))).all()
    with pytest.raises(ValueError, match="finite"):
        MorphProposal.fit(training, groups=[])


def test_morph_refit_returns_new_object_and_preserves_original_density() -> None:
    original_training = np.random.default_rng(51).normal(size=(120, 1))
    proposal = MorphProposal.fit(
        original_training,
        groups=[],
        param_names=("x",),
        kde_bw="silverman",
    )
    points = np.array([[-1.0], [0.0], [1.0]])
    original_values = proposal.log_prob(points)
    refit_training = np.random.default_rng(52).normal(4.0, 0.5, size=(80, 1))

    refitted = proposal.refit(refit_training)

    assert refitted is not proposal
    assert proposal.metadata.n_training == 120
    assert refitted.metadata.n_training == 80
    np.testing.assert_array_equal(proposal.log_prob(points), original_values)
    assert not np.allclose(refitted.log_prob(points), original_values)


def test_morph_adapter_rejects_bad_log_prob_shape() -> None:
    training = np.random.default_rng(6).normal(size=(100, 1))
    proposal = MorphProposal.fit(training, groups=[])
    with pytest.raises(InvalidProposalOutput, match="shape"):
        proposal.log_prob(np.zeros((2, 2)))


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_morph_adapter_batch_log_prob_matches_scalar_backend(ndim: int) -> None:
    training = np.random.default_rng(60 + ndim).normal(size=(160, ndim))
    proposal = MorphProposal.fit(
        training,
        groups=[],
        param_names=tuple(f"x{index}" for index in range(ndim)),
    )
    points = np.random.default_rng(70 + ndim).normal(size=(7, ndim))

    expected = np.asarray(
        [proposal._backend.logpdf(point) for point in points],
        dtype=float,
    )

    np.testing.assert_allclose(proposal.log_prob(points), expected, rtol=1e-12)


def test_morph_adapter_computes_tc_and_selects_groups_automatically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import morphZ

    training = np.random.default_rng(7).normal(size=(120, 2))
    temporary_paths: list[Path] = []

    def fake_compute_and_save_tc(
        samples: np.ndarray,
        *,
        names: list[str],
        n_order: int,
        out_path: str,
    ) -> np.ndarray:
        assert samples.shape == training.shape
        assert names == ["x", "y"]
        assert n_order == 2
        output_path = Path(out_path)
        temporary_paths.append(output_path)
        (output_path / "params_2-order_TC.json").write_text(
            json.dumps([[["x", "y"], 2.5]]),
            encoding="utf-8",
        )
        return np.array([[0.0, 2.5], [2.5, 0.0]])

    monkeypatch.setattr(
        morphZ.Nth_TC,
        "compute_and_save_tc",
        fake_compute_and_save_tc,
    )

    proposal = MorphProposal.fit(
        training,
        morph_type="2_group",
        param_names=("x", "y"),
    )

    assert proposal.metadata.group_source == "automatic:2_group"
    assert proposal.metadata.selected_groups == (("x", "y"),)
    assert proposal.metadata.single_parameters == ()
    assert temporary_paths
    assert not temporary_paths[0].exists()


@pytest.mark.parametrize("morph_type", ["n_group", "pair", "1_group", "3_group"])
def test_morph_adapter_validates_automatic_morph_type(morph_type: str) -> None:
    training = np.random.default_rng(8).normal(size=(100, 2))
    with pytest.raises(ValueError, match=r"morph_type|group order"):
        MorphProposal.fit(training, morph_type=morph_type)


def test_morph_adapter_rejects_multiple_grouping_inputs() -> None:
    training = np.random.default_rng(9).normal(size=(100, 2))
    with pytest.raises(ValueError, match="only one"):
        MorphProposal.fit(
            training,
            morph_type="2_group",
            groups=[],
        )
