def _canonical_dataset_name(dataset: str):
    ds = str(dataset).lower().strip()
    if ds in {"fmnist", "fashion", "fashionmnist", "fashion-mnist"}:
        return "fmnist"
    if ds.startswith("svhn"):
        return "svhn"
    if ds.startswith("cifar100"):
        return "cifar100"
    if ds.startswith("cifar10"):
        return "cifar10"
    return ds


def _norm_cfg(dataset: str):
    ds = _canonical_dataset_name(dataset)
    if ds == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        return mean, std
    if ds == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        return mean, std
    if ds == "svhn":
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        return mean, std
    if ds in ["mnist", "emnist"]:
        mean = (0.1307,)
        std = (0.3081,)
        return mean, std
    if ds == "fmnist":
        mean = (0.2860,)
        std = (0.3530,)
        return mean, std
    raise ValueError(f"Unknown dataset for normalization: {dataset}")
