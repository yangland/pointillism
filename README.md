# Pointillism

This repository contains the research implementation of Pointillism. It covers
Pointillism probing, model signatures, feature consensus, and consensus-based
client filtering for federated learning, together with the configurations used
for the paper's main experiments.

## Setup

The code has been tested with Python 3.10, PyTorch 2.6, and torchvision 0.21.
Create a clean environment before installing the dependencies.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

If your environment needs a specific CUDA-enabled PyTorch build, install `torch` and `torchvision` following the official PyTorch instructions for your CUDA version, then install the remaining packages from `requirements.txt`.

The scripts automatically download required torchvision datasets into `./data` by default. You can change the dataset location for FL runs with `--data_root <path>`.

## Case Studies

Model diversity and global response statistics:

```bash
python case_study_exp/exp8_prediction_diversity.py \
  -y configs/case_study/exp8_mnist_prediction_diversity.yaml
```

Feature consensus transfer across benign, backdoored, and cross-domain models:

```bash
python case_study_exp/exp6_feature_consensus_transfer.py \
  -y configs/case_study/exp6_mnist_backdoor_fmnist_feature_consensus.yaml
```

## Federated Learning Experiments
Reproduce main experiment, run one attack/config at a time:

```bash
python main_fl.py --cfg configs/fl/fmnist_series/fmnist_bad.yaml
python main_fl.py --cfg configs/fl/gtsrb_series/gtsrb_bad.yaml
python main_fl.py --cfg configs/fl/cifar10_series/cifar10_bad.yaml
```

The paper uses the following attack configs for each dataset where available:

- `label_perturbation`
- `reverse_grad`
- `a3fl`
- `bad`
- `dba`
- `neurotoxin`
- `scale`

Results are written under `log_fl/` at runtime. Generated logs, checkpoints,
and dataset payloads are intentionally not included in the repository.

## Ablation and Sensitivity

Ablation configs are in `configs/fl/fmnist_ablation/`.

Sensitivity configs are in `configs/fl/fmnist_sensitivity/`; the paper figure uses the 40% malicious-client settings for Label Perturbation and BadNets across Dirichlet alpha values.

Run one ablation setting:

```bash
python main_fl.py --cfg configs/fl/fmnist_ablation/fmnist_bad_full_default.yaml
```

Run one sensitivity setting:

```bash
python main_fl.py --cfg configs/fl/fmnist_sensitivity/fmnist_bad_mali040_alpha05.yaml
```

Summary helpers:

```bash
python auxiliary_scripts/make_fl_acc_asr_report.py <run_dirs...> --out <summary.csv>
python auxiliary_scripts/make_fl_ablation_report.py <run_dirs...> --out <summary.csv>
python auxiliary_scripts/make_fl_sensitivity_report.py <run_dirs...> --out <summary.csv>
```

## Tests

Run the public test suite from the repository root:

```bash
python -m pytest -q
```

### Pointillism client selection

`pointillism_fl.selection_mode` supports:

- `Consensus`: exact exhaustive search over fixed strict-majority subsets;
- `GreedyConsensus`: deterministic weakest-client peeling to the same fixed
  strict-majority size (`O(k^2)` with the default mean objective);
- `Pareto`: coverage/consensus scoring over a nested peeling path with variable
  subset sizes.

`GreedyConsensus`, `greedy_consensus`, and `greedy-consensus` are equivalent
spellings. Exact `Consensus` is practical for the default 10 clients per
round, but its majority-subset enumeration grows combinatorially with the
number of participating clients.

## Code Map

- `pointillism/pointillism_probe_search.py`: randomized structured probing, two-group rendering, Top-K/edge-case selection, and local refinement.
- `pointillism/pointillism_signature.py`: global and class-level model signatures.
- `pointillism/fl_defense.py`: feature consensus, prediction-diversity weighting, memory bank, and client filtering.
- `aggregation/pointillism_fl.py`: FL aggregation wrapper for Pointillism-based filtering.
- `case_study_exp/`: standalone case-study experiments and their shared helper code.
- `configs/case_study/`, `configs/fl/`: case-study and FL experiment configs.
- `auxiliary_scripts/`: auxiliary plotting/reporting/demo scripts that are not required for the main paper runs.
- `fl/`, `aggregation/`, `attacks/`, `data/`, `models/`, `utils/`: FL training, baselines, attacks, datasets, architectures, and utilities.
