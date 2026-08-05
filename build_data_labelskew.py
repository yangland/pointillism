from torch.utils.data import Subset
from torchvision import datasets, transforms
import numpy as np
import os
from collections import defaultdict
import collections
from typing import Tuple
from torch.utils.data import Dataset
from data_audio import build_audio_datasets


class _ClassSubsetDataset(Dataset):
    """Filter a classification dataset and remap the retained labels to 0..K-1."""

    def __init__(self, base: Dataset, class_ids):
        self.base = base
        self.class_ids = [int(class_id) for class_id in class_ids]
        if not self.class_ids:
            raise ValueError("task.class_subset must contain at least one class")
        if len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("task.class_subset must not contain duplicate classes")

        label_map = {old: new for new, old in enumerate(self.class_ids)}
        base_targets = _as_numpy_targets(base)
        self.indices = np.flatnonzero(np.isin(base_targets, self.class_ids)).astype(np.int64).tolist()
        self.targets = [label_map[int(base_targets[index])] for index in self.indices]
        self._label_map = label_map

        base_classes = getattr(base, "classes", None)
        if base_classes is not None:
            self.classes = [base_classes[class_id] for class_id in self.class_ids]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, old_label = self.base[self.indices[int(index)]]
        return x, self._label_map[int(old_label)]


def _apply_class_subset(train: Dataset, test: Dataset, class_ids, dataset_num_classes: int):
    class_ids = [int(class_id) for class_id in class_ids]
    invalid = [class_id for class_id in class_ids if not 0 <= class_id < int(dataset_num_classes)]
    if invalid:
        raise ValueError(
            f"task.class_subset contains invalid class ids {invalid}; "
            f"dataset has classes 0..{int(dataset_num_classes) - 1}"
        )
    return (
        _ClassSubsetDataset(train, class_ids),
        _ClassSubsetDataset(test, class_ids),
        len(class_ids),
    )


def build_label_skew_clients(cfg, data_root, rng):
    """
    Modes:
      - iid
      - fixed_num_classes
      - dirichlet
      - mixed_iid_dirichlet
      - per_label_counts (optional, if you add it)
    """
    ds_name = cfg["task"]["dataset"]

    cs = cfg["clients_setting"]
    mode = str(cs.get("non_iid_mode", "iid")).lower().strip()   # <-- default IID
    num_clients = int(cs["clients"])

    non_iid_cfg = cs.get("non_iid", {}) or {}
    labels_per_client = int(non_iid_cfg.get("labels_per_client", 10))
    dirichlet_alpha = float(non_iid_cfg.get("dirichlet_alpha", 1.0))

    allowed = {
        "iid",
        "fixed_num_classes",
        "dirichlet",
        "mixed_iid_dirichlet",
        "per_label_counts",
    }
    if mode not in allowed:
        raise ValueError(
            f"Unknown clients_setting.non_iid_mode='{mode}'. "
            f"Allowed: {sorted(allowed)}"
        )

    task_cfg = cfg.get("task", {}) or {}
    is_audio = (
        str(task_cfg.get("type", "")).lower() == "audio"
        or str(ds_name).lower() in {"speech_commands", "speechcommands", "gsc"}
    )
    if is_audio:
        train, test, num_classes = build_audio_datasets(cfg, data_root)
    else:
        img_size = task_cfg.get("img_size", [32, 32])
        if len(img_size) != 2:
            raise ValueError(f"task.img_size must be [H, W], got {img_size}")
        H, W = int(img_size[0]), int(img_size[1])
        transform = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
        ])
        train, test, num_classes = _load_dataset(
            ds_name=ds_name,
            data_root=data_root,
            transform=transform,
        )
        class_subset = task_cfg.get("class_subset", None)
        if class_subset is not None:
            train, test, num_classes = _apply_class_subset(
                train,
                test,
                class_subset,
                dataset_num_classes=num_classes,
            )

    configured_num_classes = int((cfg.get("model", {}) or {}).get("num_classes", num_classes))
    if configured_num_classes != int(num_classes):
        raise ValueError(
            f"model.num_classes={configured_num_classes} does not match the "
            f"effective dataset class count {int(num_classes)}"
        )
    y_all = _as_numpy_targets(train)

    # ---------------- iid ----------------
    if mode == "iid":
        idxs = list(range(len(train)))
        rng.shuffle(idxs)
        base = len(idxs) // num_clients
        rem = len(idxs) % num_clients

        client_sets = {}
        client_label_counts = {}
        start = 0
        for cid in range(num_clients):
            sz = base + (1 if cid < rem else 0)
            client_idxs = idxs[start:start + sz]
            client_sets[cid] = Subset(train, client_idxs)
            client_label_counts[cid] = np.bincount(y_all[np.asarray(client_idxs, dtype=np.int64)], minlength=num_classes).astype(int).tolist()
            start += sz

        aux = {"train_union": train, "client_label_counts": client_label_counts}
        return client_sets, test, aux

    # ---------------- dirichlet ----------------
    elif mode == "dirichlet":
        per_participant = sample_dirichlet_train_data(
            train_dataset=train,
            no_participants=num_clients,
            alpha=dirichlet_alpha,
            seed=rng,
        )
        client_sets = {u: Subset(train, per_participant[u]) for u in range(num_clients)}
        client_label_counts = {
            cid: np.bincount(y_all[np.asarray(sub.indices, dtype=np.int64)], minlength=num_classes).astype(int).tolist()
            for cid, sub in client_sets.items()
        }
        aux = {"train_union": train, "client_label_counts": client_label_counts}
        return client_sets, test, aux

    # ---------------- mixed IID + Dirichlet ----------------
    elif mode == "mixed_iid_dirichlet":
        num_non_iid = int(non_iid_cfg.get("non_iid_clients", -1))
        num_iid = int(non_iid_cfg.get("iid_clients", -1))
        if num_non_iid <= 0 or num_iid <= 0:
            raise ValueError(
                "mixed_iid_dirichlet requires positive "
                "clients_setting.non_iid.non_iid_clients and iid_clients"
            )
        if num_non_iid + num_iid != num_clients:
            raise ValueError(
                "mixed_iid_dirichlet client counts must sum to "
                f"clients_setting.clients={num_clients}, got "
                f"{num_non_iid}+{num_iid}"
            )

        # Malicious assignment is deterministic (the first round(mali_rate*C)
        # client IDs), so keeping the Dirichlet clients first guarantees that
        # every malicious client uses the same non-IID distribution as the
        # benign non-IID clients.
        num_malicious = int(round(float(cs.get("mali_rate", 0.0)) * num_clients))
        if num_malicious > num_non_iid:
            raise ValueError(
                f"mixed_iid_dirichlet has {num_malicious} malicious clients but "
                f"only {num_non_iid} non-IID client slots"
            )

        per_participant, partition_modes = sample_mixed_iid_dirichlet_train_data(
            train_dataset=train,
            num_non_iid_clients=num_non_iid,
            num_iid_clients=num_iid,
            alpha=dirichlet_alpha,
            seed=rng,
        )
        client_sets = {
            cid: Subset(train, per_participant[cid])
            for cid in range(num_clients)
        }
        client_label_counts = {
            cid: np.bincount(
                y_all[np.asarray(sub.indices, dtype=np.int64)],
                minlength=num_classes,
            ).astype(int).tolist()
            for cid, sub in client_sets.items()
        }
        aux = {
            "train_union": train,
            "client_label_counts": client_label_counts,
            "client_partition_modes": partition_modes,
        }
        return client_sets, test, aux

    # ---------------- fixed_num_classes (only if explicitly selected) ----------------
    elif mode == "fixed_num_classes":
        # ---- your existing fixed_num_classes block exactly as-is ----
        # 2) Per-class index lists
        labels = _as_numpy_targets(train)
        class_idxs = collections.defaultdict(list)
        for idx, y in enumerate(labels):
            class_idxs[int(y)].append(idx)

        for c in range(num_classes):
            rng.shuffle(class_idxs[c])

        # 3) Draw labels for each client
        client_labels_map = collections.defaultdict(list)
        classes = list(range(num_classes))
        all_class_assignment = collections.defaultdict(list)

        for cid in range(num_clients):
            chosen_labels = rng.choice(classes, size=labels_per_client, replace=False).tolist()
            client_labels_map[cid].extend(chosen_labels)
            for c in chosen_labels:
                all_class_assignment[c].append(cid)

        # 4) Allocate disjoint chunks per class across assigned clients
        client_indices = collections.defaultdict(list)
        for c in classes:
            all_idxs = class_idxs[c]
            cids = all_class_assignment[c]
            if not cids:
                continue
            num_clients_c = len(cids)
            num_samples_c = len(all_idxs)

            split_sizes = [num_samples_c // num_clients_c] * num_clients_c
            remainder = num_samples_c % num_clients_c
            for i in range(remainder):
                split_sizes[i] += 1

            start = 0
            for i, cid in enumerate(cids):
                end = start + split_sizes[i]
                if end > start:
                    client_indices[cid].extend(all_idxs[start:end])
                start = end

        client_sets = {cid: Subset(train, client_indices[cid]) for cid in range(num_clients)}
        client_label_counts = {
            cid: np.bincount(y_all[np.asarray(idxs, dtype=np.int64)], minlength=num_classes).astype(int).tolist()
            for cid, idxs in client_indices.items()
        }
        for cid in range(num_clients):
            client_label_counts.setdefault(cid, [0] * num_classes)
        aux = {"train_union": train, "client_label_counts": client_label_counts}
        return client_sets, test, aux

    else:
        raise NotImplementedError("per_label_counts mode not implemented yet")


def _load_dataset(
    ds_name: str,
    data_root: str,
    transform,
) -> Tuple[Dataset, Dataset, int]:
    """
    Return (train_set, test_set, num_classes) for supported datasets.
    Supports:
      - MNIST
      - FashionMNIST
      - CIFAR10
      - SVHN
      - CIFAR100
      - GTSRB
      - Imagenette / Imagenette160 / Imagenette320
    """
    name = ds_name.lower()

    if name.startswith("mnist"):
        train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        test  = datasets.MNIST(data_root, train=False, download=True, transform=transform)
        num_classes = 10

    elif name.startswith("fmnist") or name.startswith("fashion"):
        train = datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)
        test  = datasets.FashionMNIST(data_root, train=False, download=True, transform=transform)
        num_classes = 10

    # Check CIFAR-100 before CIFAR-10: "cifar100" also starts with "cifar10".
    elif name.startswith("cifar100"):
        train = datasets.CIFAR100(data_root, train=True, download=True, transform=transform)
        test  = datasets.CIFAR100(data_root, train=False, download=True, transform=transform)
        num_classes = 100

    elif name.startswith("cifar10"):
        train = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
        test  = datasets.CIFAR10(data_root, train=False, download=True, transform=transform)
        num_classes = 10

    elif name.startswith("svhn"):
        train = datasets.SVHN(data_root, split="train", download=True, transform=transform)
        test  = datasets.SVHN(data_root, split="test", download=True, transform=transform)
        num_classes = 10

    elif name.startswith("gtsrb"):
        # German Traffic Sign Recognition Benchmark, 43 classes
        train = datasets.GTSRB(data_root, split="train", download=True, transform=transform)
        test  = datasets.GTSRB(data_root, split="test",  download=True, transform=transform)
        num_classes = 43

    elif name.startswith("imagenette"):
        # Support imagenette, imagenette160, imagenette320
        size = "full"
        if "160" in name:
            size = "160px"
        elif "320" in name:
            size = "320px"

        train = datasets.Imagenette(
            data_root,
            split="train",
            size=size,
            download=True,
            transform=transform,
        )
        test = datasets.Imagenette(
            data_root,
            split="val",
            size=size,
            download=True,
            transform=transform,
        )
        num_classes = 10

    else:
        raise ValueError(f"Unsupported dataset for label-skew builder: {ds_name}")

    return train, test, num_classes


def sample_dirichlet_train_data(train_dataset, no_participants, alpha=0.5, seed=None):
    """
    Balanced Dirichlet label split.
    - Uses Dirichlet(alpha) per class to shape label proportions.
    - Enforces near-equal total sizes per client (N//K or N//K+1).
    - Uses all samples.
    - Allows zero samples for any (client, label).

    Returns: dict[user] -> list of sample indices
    """

    labels = _as_numpy_targets(train_dataset)  # np.int64 [N]
    N = int(labels.shape[0])
    K = int(no_participants)
    classes = np.unique(labels)

    # Per-class index lists
    cls_indices = {c: np.where(labels == c)[0] for c in classes}
    rng = np.random.default_rng(seed)

    # Target capacities: as equal as possible, exact sum = N
    base = N // K
    remainder = N % K
    # First `remainder` clients get one more sample
    target_cap = np.full(K, base, dtype=int)
    if remainder:
        target_cap[:remainder] += 1
    # Shuffle which clients get the +1 so it’s not always the first few
    rng.shuffle(target_cap)

    # Running capacities (how many more samples each client can take)
    cap = target_cap.copy()

    # Where we will accumulate results
    per_participant = {u: [] for u in range(K)}

    for c in classes:
        idx = cls_indices[c].copy()
        rng.shuffle(idx)
        m = len(idx)
        if m == 0:
            continue

        # Dirichlet proportions for this class
        probs = rng.dirichlet(np.full(K, float(alpha), dtype=np.float64))

        # NEW: bias by remaining capacity so we don't exhaust some clients early
        cap_sum = cap.sum()
        if cap_sum == 0:
            continue  # no room left anywhere (shouldn't happen if targets sum to N)
        weighted = probs * cap
        if weighted.sum() == 0:
            # fall back to capacity-only if probs are extremely skewed numerically
            weighted = cap.astype(float)

        weighted /= weighted.sum()

        # Use weighted proportions (not raw probs) to propose sizes
        sizes = np.floor(weighted * m).astype(int)

        # Respect remaining capacity
        sizes = np.minimum(sizes, cap)
        cap -= sizes
        cap = np.maximum(cap, 0)

        assigned = sizes.sum()
        leftover = m - assigned

        # Distribute leftovers by the same weighted scores times remaining cap
        while leftover > 0 and cap.sum() > 0:
            score = weighted * cap
            if not np.any(score):
                score = cap.astype(float)
            u = int(np.argmax(score))
            sizes[u] += 1
            cap[u]  -= 1
            leftover -= 1


        # Now slice the class indices according to `sizes`
        # We do a stable split in client order; re-ordering by capacity isn’t necessary
        cuts = np.cumsum(sizes)[:-1]
        chunks = np.split(idx, cuts)
        for u, chunk in enumerate(chunks):
            if chunk.size:
                per_participant[u].extend(chunk.tolist())

    # Sanity checks (optional; remove in production if you want)
    # 1) All data used
    used = sum(len(v) for v in per_participant.values())
    assert used == N, f"Assignment lost samples: used {used} vs N {N}"

    # 2) Totals are equalized to target_cap
    # Some tiny drift could occur if classes are empty; correct by moving a few indices.
    totals = np.array([len(per_participant[u]) for u in range(K)], dtype=int)
    drift = totals - target_cap
    if np.any(drift):
        # move minimal number of samples to fix drift
        over = list(np.where(drift > 0)[0])
        under = list(np.where(drift < 0)[0])
        # Simple balancing: pop from overfull and push to underfull
        oi = ui = 0
        while oi < len(over) and ui < len(under):
            o = over[oi]; u = under[ui]
            move_cnt = min(drift[o], -drift[u])
            # move last `move_cnt` samples
            moved = per_participant[o][-move_cnt:]
            per_participant[o] = per_participant[o][:-move_cnt]
            per_participant[u].extend(moved)
            drift[o] -= move_cnt
            drift[u] += move_cnt
            if drift[o] == 0: oi += 1
            if drift[u] == 0: ui += 1

    return per_participant


def sample_mixed_iid_dirichlet_train_data(
    train_dataset,
    num_non_iid_clients,
    num_iid_clients,
    alpha=0.5,
    seed=None,
):
    """
    Build disjoint, exhaustive, balanced client partitions for a mixed
    population.

    Client IDs ``0..num_non_iid_clients-1`` receive a balanced Dirichlet
    partition. Remaining client IDs receive an IID partition. The dataset is
    first split into class-stratified group pools so both groups see the same
    overall class mixture while every training example is assigned exactly
    once.

    Returns:
      - dict[cid] -> global sample indices
      - dict[cid] -> ``"dirichlet"`` or ``"iid"``
    """
    num_non_iid_clients = int(num_non_iid_clients)
    num_iid_clients = int(num_iid_clients)
    if num_non_iid_clients <= 0 or num_iid_clients <= 0:
        raise ValueError("Mixed partition requires at least one client in each group")
    if float(alpha) <= 0.0:
        raise ValueError(f"Dirichlet alpha must be positive, got {alpha}")

    rng = np.random.default_rng(seed)
    labels = _as_numpy_targets(train_dataset)
    num_samples = int(labels.shape[0])
    num_clients = num_non_iid_clients + num_iid_clients

    # Match the ordinary balanced split: clients have N//C samples, with the
    # first N%C slots receiving one additional sample.
    base = num_samples // num_clients
    remainder = num_samples % num_clients
    non_iid_pool_size = (
        base * num_non_iid_clients + min(remainder, num_non_iid_clients)
    )

    classes, class_counts = np.unique(labels, return_counts=True)
    ideal = class_counts.astype(np.float64) * (
        float(non_iid_pool_size) / float(max(num_samples, 1))
    )
    non_iid_class_counts = np.floor(ideal).astype(np.int64)
    leftover = int(non_iid_pool_size - int(non_iid_class_counts.sum()))
    if leftover:
        # Random tie-breaking prevents low class IDs from always receiving the
        # rounding remainder while retaining deterministic seeded behavior.
        tie_break = rng.random(len(classes))
        order = np.lexsort((tie_break, -(ideal - non_iid_class_counts)))
        non_iid_class_counts[order[:leftover]] += 1

    non_iid_pool = []
    iid_pool = []
    for class_id, take in zip(classes, non_iid_class_counts):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        split_at = int(take)
        non_iid_pool.extend(class_indices[:split_at].tolist())
        iid_pool.extend(class_indices[split_at:].tolist())

    rng.shuffle(non_iid_pool)
    rng.shuffle(iid_pool)
    if len(non_iid_pool) != non_iid_pool_size:
        raise RuntimeError(
            f"Mixed partition built {len(non_iid_pool)} non-IID pool samples; "
            f"expected {non_iid_pool_size}"
        )

    # sample_dirichlet_train_data returns indices local to this Subset.
    non_iid_subset = Subset(train_dataset, non_iid_pool)
    local_non_iid = sample_dirichlet_train_data(
        train_dataset=non_iid_subset,
        no_participants=num_non_iid_clients,
        alpha=alpha,
        seed=rng,
    )

    per_participant = {}
    partition_modes = {}
    non_iid_pool_array = np.asarray(non_iid_pool, dtype=np.int64)
    for cid in range(num_non_iid_clients):
        local_indices = np.asarray(local_non_iid[cid], dtype=np.int64)
        per_participant[cid] = non_iid_pool_array[local_indices].tolist()
        partition_modes[cid] = "dirichlet"

    iid_base = len(iid_pool) // num_iid_clients
    iid_remainder = len(iid_pool) % num_iid_clients
    start = 0
    for offset in range(num_iid_clients):
        size = iid_base + (1 if offset < iid_remainder else 0)
        cid = num_non_iid_clients + offset
        per_participant[cid] = iid_pool[start:start + size]
        partition_modes[cid] = "iid"
        start += size

    assigned = [
        sample_idx
        for cid in range(num_clients)
        for sample_idx in per_participant[cid]
    ]
    if len(assigned) != num_samples or len(set(assigned)) != num_samples:
        raise RuntimeError(
            "Mixed IID/Dirichlet partition must assign every sample exactly once"
        )
    return per_participant, partition_modes


def _as_numpy_targets(train_dataset):
    """
    Return labels as a 1D np.int64 array, supporting:
      - Standard torchvision datasets with .targets or .labels
      - torch.utils.data.Subset
      - torchvision.datasets.GTSRB, which stores labels in ._samples
    """
    ds = train_dataset

    # Case 1: Subset -> recurse on underlying dataset and index
    if isinstance(ds, Subset):
        base = ds.dataset
        base_targets = _as_numpy_targets(base)
        idx = np.asarray(ds.indices, dtype=np.int64)
        return base_targets[idx]

    # Case 2: usual torchvision datasets (CIFAR, MNIST, ImageFolder, etc.)
    if hasattr(ds, "targets"):
        return np.asarray(ds.targets, dtype=np.int64)
    if hasattr(ds, "labels"):
        return np.asarray(ds.labels, dtype=np.int64)

    # Case 3: torchvision GTSRB (and similar) -> labels in ._samples
    if hasattr(ds, "_samples"):
        samples = ds._samples  # list of (path, label)
        try:
            labels = [y for _, y in samples]
        except Exception as e:
            raise ValueError(
                "Dataset has _samples but unpacking (path, label) failed; "
                "adjust _as_numpy_targets for this dataset."
            ) from e
        return np.asarray(labels, dtype=np.int64)

    # If you hit this, add another branch for your custom dataset
    raise ValueError("train has no .targets/.labels/._samples; "
                     "extend _as_numpy_targets for this dataset type")



def log_client_label_counts(client_sets, train, num_classes, out_dir=None, prefix="[non_iid]", logger=None):
    """
    Print or log per-client label counts and optionally save a CSV.

    Args:
        client_sets: dict[cid -> Subset(train, indices)]
        train: base dataset whose targets are read by _as_numpy_targets(train)
        num_classes: int
        out_dir: optional directory path to save client_label_counts.csv
        prefix: message prefix string
        logger: optional logger; uses logger.info(...) when provided, else print(...)
    """
    def _log(msg: str):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    y_all = _as_numpy_targets(train)
    rows = []
    for cid, sub in sorted(client_sets.items()):
        assert hasattr(sub, "indices"), f"client {cid} dataset is not a Subset"
        idxs = np.asarray(sub.indices, dtype=int)
        y = y_all[idxs]
        counts = np.bincount(y, minlength=num_classes)
        msg = ", ".join(f"{c}:{int(counts[c])}" for c in range(num_classes) if counts[c] > 0)
        _log(f"{prefix} client {cid}: {msg}")
        rows.append([cid] + [int(x) for x in counts])

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        header = ",".join(["client"] + [f"class_{c}" for c in range(num_classes)])
        path = os.path.join(out_dir, "client_label_counts.csv")
        with open(path, "w") as f:
            f.write(header + "\n")
            for r in rows:
                f.write(",".join(map(str, r)) + "\n")
        _log(f"{prefix} saved -> {path}")
