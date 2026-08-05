import numpy as np
from torch.utils.data import Dataset

import build_data_labelskew as data_builder


class _SyntheticDataset(Dataset):
    def __init__(self, targets, classes=None):
        self.targets = list(targets)
        self.classes = list(classes or [])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return f"image-{index}", self.targets[index]


def test_cifar100_dispatch_is_not_shadowed_by_cifar10(monkeypatch):
    calls = []

    def fake_cifar10(*args, **kwargs):
        calls.append("cifar10")
        return _SyntheticDataset([0])

    def fake_cifar100(*args, **kwargs):
        calls.append("cifar100")
        return _SyntheticDataset([0])

    monkeypatch.setattr(data_builder.datasets, "CIFAR10", fake_cifar10)
    monkeypatch.setattr(data_builder.datasets, "CIFAR100", fake_cifar100)

    train, test, num_classes = data_builder._load_dataset(
        ds_name="cifar100",
        data_root="/unused",
        transform=None,
    )

    assert isinstance(train, _SyntheticDataset)
    assert isinstance(test, _SyntheticDataset)
    assert num_classes == 100
    assert calls == ["cifar100", "cifar100"]


def test_class_subset_filters_and_remaps_in_declared_order():
    classes = [f"class-{index}" for index in range(100)]
    base = _SyntheticDataset(
        targets=[97, 3, 42, 43, 88, 1, 3],
        classes=classes,
    )

    subset = data_builder._ClassSubsetDataset(base, [3, 42, 43, 88, 97])

    assert len(subset) == 6
    assert subset.classes == [
        "class-3",
        "class-42",
        "class-43",
        "class-88",
        "class-97",
    ]
    assert subset.targets == [4, 0, 1, 2, 3, 0]
    assert np.bincount(subset.targets, minlength=5).tolist() == [2, 1, 1, 1, 1]
    assert [subset[index][1] for index in range(len(subset))] == subset.targets


def test_mixed_iid_dirichlet_partition_is_balanced_disjoint_and_seeded():
    dataset = _SyntheticDataset(
        targets=[
            class_id
            for class_id in range(10)
            for _ in range(100)
        ]
    )

    first, first_modes = data_builder.sample_mixed_iid_dirichlet_train_data(
        dataset,
        num_non_iid_clients=65,
        num_iid_clients=35,
        alpha=0.5,
        seed=2026,
    )
    second, second_modes = data_builder.sample_mixed_iid_dirichlet_train_data(
        dataset,
        num_non_iid_clients=65,
        num_iid_clients=35,
        alpha=0.5,
        seed=2026,
    )

    assert first == second
    assert first_modes == second_modes
    assert set(first) == set(range(100))
    assert all(len(indices) == 10 for indices in first.values())
    assert {first_modes[cid] for cid in range(65)} == {"dirichlet"}
    assert {first_modes[cid] for cid in range(65, 100)} == {"iid"}

    assigned = [index for indices in first.values() for index in indices]
    assert sorted(assigned) == list(range(len(dataset)))

    non_iid_indices = [
        index for cid in range(65) for index in first[cid]
    ]
    non_iid_labels = np.asarray(dataset.targets)[non_iid_indices]
    assert np.bincount(non_iid_labels, minlength=10).tolist() == [65] * 10


def test_mixed_mode_rejects_malicious_clients_outside_non_iid_pool(monkeypatch):
    dataset = _SyntheticDataset(
        targets=[class_id for class_id in range(10) for _ in range(10)]
    )
    monkeypatch.setattr(
        data_builder,
        "_load_dataset",
        lambda **kwargs: (dataset, dataset, 10),
    )
    cfg = {
        "task": {"dataset": "cifar10", "img_size": [32, 32]},
        "model": {"num_classes": 10},
        "clients_setting": {
            "clients": 10,
            "mali_rate": 0.4,
            "non_iid_mode": "mixed_iid_dirichlet",
            "non_iid": {
                "dirichlet_alpha": 0.5,
                "non_iid_clients": 3,
                "iid_clients": 7,
            },
        },
    }

    with np.testing.assert_raises_regex(
        ValueError, "4 malicious clients.*only 3 non-IID"
    ):
        data_builder.build_label_skew_clients(
            cfg,
            data_root="/unused",
            rng=np.random.default_rng(1),
        )
