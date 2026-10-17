# Sequential Matching: ML_Wizards_IITM, Round 2 build

A decision policy for the Vouchsafe / Romeo & Juliet Sequential Matching Problem (release 1.0.0). Each simulated day it chooses whose deal-breakers to ask about and which feasible, non-overlapping pairs to introduce. Inference runs offline on CPU, in Python's standard library only.

| Document | Content |
|---|---|
| **[BEST_SOLUTION.md](BEST_SOLUTION.md)** | **The technical report.** The shipped policy stage by stage, results against all three baselines on all six families, inference time, head-room, failure cases and limitations |
| [EXPERIMENTS_BY_STAGE.md](EXPERIMENTS_BY_STAGE.md) | Every experiment and ablation, with data, paired intervals and decisions |
| [ASK_STRATEGY.md](ASK_STRATEGY.md) | The Bayesian clarification policy, with a worked example |
| [docs/STARTER_README.md](docs/STARTER_README.md) | The organisers' original kit README |

## The policy in brief

1. **Asking.** Only hard-constraint bundles are asked. Askable members are ordered by Bayesian expected number of compatible partners. Anyone under a 5% chance of any partner is skipped for that day and re-scored daily.
2. **Matching.** The organisers' `eligibility()` is the only gate. A 14-feature logistic yes-model ranks pairs. An exact maximum-weight matching (Edmonds' blossom algorithm) uses tiered weights in this order: most introductions, then most first-time members, then scarce members, then predicted value.
3. **Repeat.** The policy keeps no memory and rebuilds everything from each observation. Nobody with a feasible partner is held back. Every ask and pair is re-validated, with fallbacks to the reference rule and then to empty lists.

## Setup

Inference and evaluation need **Python 3.10 or later and nothing else**. The policy image uses `python:3.12.7-slim`, and CI runs the tests on Python 3.10 and 3.12. Development used Python 3.14.6.

Training and the analysis scripts additionally need the libraries in `requirements-analysis.txt`. They are never imported by the policy and are not in the image.

```bash
pip install -r requirements-analysis.txt      # only for training and analysis, not for inference
```

On Windows, clone with `git clone -c core.autocrlf=false <repo>` so that `verify_data.py` sees the exact data bytes.

## Exact commands

**Checks (the organisers' sequence):**

```bash
python -m unittest -v                          # 42 tests: kit, release, runner, policy safety, matcher
python verify_data.py
docker build -t sequential-policy:submission .
python evaluate.py --image sequential-policy:submission --seeds 101 --variants all --output results/container_check.json
```

In local mode `evaluate.py` always passes `--baseline` (default `greedy`) to the policy, so it evaluates the reference baselines. Our policy is what the container runs with no arguments, and what `run_final_study.py` runs locally.

**The four-way comparison through the organisers' harness** (`evaluate.episode`, one process per call, seeds 101–120 × 6 families):

```bash
python run_final_study.py --seeds 101-120 --output-dir results/official
```

**The larger paired studies** (in-process, independent worlds per family):

```bash
python experiments.py --configs greedy,no_asks,random,final,tiered,submission --seeds 6001-6060 --independent --output results/study_final.json
python experiments.py --configs greedy,final,tiered,tiered_nocov,tiered_novalue,exact_value --seeds 2001-2060 --independent --output results/study_r2a.json
python experiments.py --configs tiered,tiered_valueask,tiered_bayes,tiered_bayes_skip --seeds 4001-4060 --independent --output results/study_asks.json
python experiments.py --configs submission_v1,sub_skip10,sub_scarce,sub_adapt --seeds 5001-5060 --independent --output results/study_next.json
python experiments.py --configs submission,wait_v_p25,wait_v_p50,wait_v_p50_r40,wait_soft1,wait_soft2 --seeds 8001-8060 --independent --output results/study_wait.json
python ask_model_study.py --seeds 3001-3020                     # Bayesian ask model: prediction quality
python soft_status_study.py --seeds 7001-7005                   # soft answers at introduction; pair-value percentiles
python headroom_r2.py --seeds 4001-4030                         # coverage and MSMI maximums (diagnostic)
```

`results/study_next.json` was produced when the default had no scarcity bonus. That configuration is now named `submission_v1`, and the stored file labels it `submission`.

**Training the shipped model:**

```bash
python experiments.py --configs greedy,random,global_learned --seeds 1001-1040 --variants all --independent --collect --output results/train_rows_r2.npz
python train_shipped.py                                         # writes model_weights.json and results/offline_metrics_r2.json
```

## Seeds and randomness

| Item | Value |
|---|---|
| Training rollouts | Seeds 1001–1040 × 6 families. World seed = seed + 100000 × family index (development 0, sparse 1, cold start 2, delayed 3, shift 4, drift 5). Behaviour policies: greedy, random feasible, `global_learned` |
| Model validation split | Seeds ≤ 1030 train, ≥ 1031 held out; the shipped weights are then refitted on all 40 seeds |
| History priors | Supplied training pools `public_01`–`public_06`: reply rate 0.797, yes rate 0.479 |
| **Inference random seed** | The policy is deterministic given the request. The only pseudo-random step, the Bayesian ask model, uses `random.Random('bayes-asks|<day>')`. The released random-feasible baseline uses `random.Random(17 + day)` |
| Experiment seeds | Listed with each command above; every study uses fresh seeds |

## Inference assets and limits

- **Image contents:** `kit.py`, `policy.py`, `smp/` and `model_weights.json` (2.8 KiB). `.dockerignore` excludes everything else.
- **No network, no GPU, no external services.** Memory between calls is `{}`.
- **Speed:** about 0.2 s per call including process start-up (measured through the organisers' harness), against a 10 s limit. An internal 6 s budget switches to a greedy matcher if time runs short.

## Results files

All are machine-readable and committed, although `results/` is otherwise ignored.

| File | Content |
|---|---|
| `results/official/{greedy,no_asks,random,round1_final,submission}.json` | Full-episode results in `evaluate.py` format: seeds 101–120, all six families, the organisers' harness |
| `results/study_final.json` | Shipped policy vs the three baselines and earlier versions, 2,160 episodes |
| `results/study_r2a.json`, `study_asks.json`, `study_next.json`, `study_wait.json` | Ablations: matcher tiers, ask order and skipping, scarcity, re-learning, waiting thresholds |
| `results/headroom_r2.json` | Coverage ceiling, MSMI upper bound and luck-free expected MSMI (diagnostic) |
| `results/ask_model_study.json`, `results/soft_status_study.json` | Bayesian ask-model accuracy; soft-answer status at introduction |
| `results/train_rows_r2.npz`, `results/offline_metrics_r2.json`, `results/train_shipped.log` | Training data, held-out metrics and the training log of the shipped model |

## Provenance and licences

| Item | Declaration |
|---|---|
| Models | One logistic regression trained by us (`train_shipped.py`); no pretrained or external models |
| Data | Only the supplied synthetic kit data and our own simulator rollouts; no external or personal data |
| Inference dependencies | Python standard library only (PSF licence) |
| Training and analysis libraries | NumPy 2.5.1, SciPy 1.18.1, scikit-learn 1.9.0 (all BSD-3-Clause); never part of the image |
| Training compute | One laptop CPU (12 threads), minutes per training run; no GPU |
| Starter kit and data | Organisers' terms: [LICENSE](LICENSE), [DATA_LICENSE.md](DATA_LICENSE.md) |
| Coding tools | Claude (Anthropic) for analysis review, report drafting and submission checks. All code and results were reviewed and run by the team. |
| Licence for our own code | MIT, © 2026 ML_Wizards_IITM (`smp/`, `policy.py`, training and analysis scripts). The organisers' kit files and data remain under [LICENSE](LICENSE) and [DATA_LICENSE.md](DATA_LICENSE.md). |
