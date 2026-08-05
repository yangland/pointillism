import math

import numpy as np

from pointillism.fl_defense import (
    PointillismFLState,
    _stable_mask_seed,
    build_round_probe_pool,
)
from pointillism.pointillism_probe_search import (
    sample_random_masks_for_d_batched_cpu,
    sample_two_group_color_centers_batch_cpu,
)


def test_vectorized_color_centers_are_deterministic_and_valid():
    seeds = [1, 2, 3, 2, 2147483646]
    fg_a, bg_a = sample_two_group_color_centers_batch_cpu(
        seeds=seeds,
        d_min=0.4,
        d_max=math.sqrt(3.0),
        num_channels=3,
    )
    fg_b, bg_b = sample_two_group_color_centers_batch_cpu(
        seeds=seeds,
        d_min=0.4,
        d_max=math.sqrt(3.0),
        num_channels=3,
    )

    assert fg_a.shape == (len(seeds), 3)
    assert bg_a.shape == (len(seeds), 3)
    assert np.array_equal(fg_a, fg_b)
    assert np.array_equal(bg_a, bg_b)
    assert np.array_equal(fg_a[1], fg_a[3])
    assert np.array_equal(bg_a[1], bg_a[3])
    assert float(fg_a.min()) >= 0.0
    assert float(fg_a.max()) <= 1.0
    assert float(bg_a.min()) >= 0.0
    assert float(bg_a.max()) <= 1.0

    distances = np.linalg.norm(
        fg_a.astype(np.float64) - bg_a.astype(np.float64),
        axis=1,
    )
    assert np.all(distances >= 0.4 - 1e-6)
    assert np.all(distances <= math.sqrt(3.0) + 1e-6)


def test_batched_mask_sampler_exhausts_small_spaces_without_duplicates():
    records = sample_random_masks_for_d_batched_cpu(
        grid_hw=4,
        w_list=[1, 3],
        n_trials_per_w=2000,
        seed=6060,
        max_records=8000,
        batch_size=64,
    )

    by_weight = {
        weight: [tuple(record["flat_idx"]) for record in records if record["w"] == weight]
        for weight in (1, 3)
    }
    assert len(by_weight[1]) == math.comb(16, 1)
    assert len(by_weight[3]) == math.comb(16, 3)
    assert len(set(by_weight[1])) == len(by_weight[1])
    assert len(set(by_weight[3])) == len(by_weight[3])
    assert all(
        record["mask_seed"] == _stable_mask_seed(record["flat_idx"])
        for record in records
    )


def test_vectorized_pool_builder_preserves_workload_and_is_deterministic():
    cfg = {
        "seed": 6060,
        "task": {"dataset": "cifar10", "img_size": [32, 32], "channels": 3},
        "model": {"num_classes": 10},
    }
    cfg_point = {
        "probe_mode": "two_group",
        "pool_construction_backend": "cpu_vectorized",
        "grid_hws": [4, 8],
        "random_budget_per_grid": 32,
        "reuse_random_pool_frac": 0.0,
        "use_memory_bank": False,
        "color_inertia": 0.0,
        "color_group": {
            "variants_per_mask": 1,
            "sigma_fg": 0.05,
            "sigma_bg": 0.05,
            "d_min": 0.4,
        },
        "w_schedule": {
            "mode": "pct",
            "pcts": [0.08, 0.16, 0.32, 0.5],
            "use_complement": False,
            "rounding": "nearest",
            "clamp": [1, -1],
            "dedup_w": True,
        },
    }

    specs_a, meta_a = build_round_probe_pool(
        cfg=cfg,
        cfg_point=cfg_point,
        state=PointillismFLState(),
        rnd=0,
    )
    specs_b, meta_b = build_round_probe_pool(
        cfg=cfg,
        cfg_point=cfg_point,
        state=PointillismFLState(),
        rnd=0,
    )

    assert len(specs_a) == 64
    assert meta_a == meta_b
    assert meta_a["construction_backend"] == "cpu_vectorized"
    assert specs_a == specs_b
    assert all(len(spec["mu_fg"]) == 3 for spec in specs_a)
    assert all(len(spec["mu_bg"]) == 3 for spec in specs_a)
