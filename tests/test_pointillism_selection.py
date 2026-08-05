from __future__ import annotations

from unittest import mock

import numpy as np

from pointillism.fl_defense import select_client_subset


def _client_infos(num_clients: int, num_classes: int = 3):
    labels = list(range(num_classes))
    return [
        {
            "cid": idx,
            "q1": {"effective_support_labels": labels},
            "signature": {},
        }
        for idx in range(num_clients)
    ]


def _cfg(mode: str, objective: str = "mean"):
    return {
        "selection_mode": mode,
        "min_keep_frac": 0.5,
        "consensus_objective": objective,
        "consensus_quantile_alpha": 0.5,
        "lambda_cov": 0.35,
        "lambda_con": 0.65,
    }


def test_greedy_consensus_matches_exact_consensus_on_separated_groups():
    num_clients = 10
    benign = set(range(6))
    agreement = np.zeros((num_clients, num_clients), dtype=np.float32)
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            value = 0.9 if i in benign and j in benign else 0.1
            agreement[i, j] = agreement[j, i] = value

    exact_keep, exact_reject, _ = select_client_subset(
        client_infos=_client_infos(num_clients),
        A=agreement,
        num_classes=3,
        cfg_point=_cfg("Consensus"),
        rng_seed=123,
    )
    greedy_keep, greedy_reject, info = select_client_subset(
        client_infos=_client_infos(num_clients),
        A=agreement,
        num_classes=3,
        cfg_point=_cfg("GreedyConsensus"),
        rng_seed=123,
    )

    assert exact_keep == greedy_keep == sorted(benign)
    assert exact_reject == greedy_reject == [6, 7, 8, 9]
    assert info["mode"] == "greedy_consensus_peeling"
    assert len(info["subset_rows"]) == 1


def test_greedy_consensus_uses_strict_majority_without_enumeration():
    num_clients = 20
    rng = np.random.default_rng(456)
    agreement = rng.random((num_clients, num_clients), dtype=np.float32)
    agreement = 0.5 * (agreement + agreement.T)
    np.fill_diagonal(agreement, 0.0)

    with mock.patch(
        "pointillism.fl_defense.itertools.combinations",
        side_effect=AssertionError("GreedyConsensus must not enumerate subsets"),
    ):
        keep, reject, info = select_client_subset(
            client_infos=_client_infos(num_clients),
            A=agreement,
            num_classes=3,
            cfg_point=_cfg("greedy_consensus"),
            rng_seed=789,
        )

    assert len(keep) == 11
    assert len(reject) == 9
    assert set(keep).isdisjoint(reject)
    assert sorted(keep + reject) == list(range(num_clients))
    assert len(info["subset_rows"]) == 1


def test_greedy_consensus_aliases_and_tie_breaking_are_deterministic():
    agreement = np.zeros((10, 10), dtype=np.float32)
    expected = None
    for mode in ("GreedyConsensus", "greedy_consensus", "greedy-consensus"):
        first = select_client_subset(
            client_infos=_client_infos(10),
            A=agreement,
            num_classes=3,
            cfg_point=_cfg(mode),
            rng_seed=101,
        )
        second = select_client_subset(
            client_infos=_client_infos(10),
            A=agreement,
            num_classes=3,
            cfg_point=_cfg(mode),
            rng_seed=101,
        )
        assert first[0] == second[0]
        expected = first[0] if expected is None else expected
        assert first[0] == expected


def test_greedy_consensus_supports_non_mean_objectives():
    rng = np.random.default_rng(2026)
    agreement = rng.random((10, 10), dtype=np.float32)
    agreement = 0.5 * (agreement + agreement.T)
    np.fill_diagonal(agreement, 0.0)

    for objective in ("median", "low_quantile"):
        keep, reject, _ = select_client_subset(
            client_infos=_client_infos(10),
            A=agreement,
            num_classes=3,
            cfg_point=_cfg("GreedyConsensus", objective),
            rng_seed=2026,
        )
        assert len(keep) == 6
        assert len(reject) == 4
