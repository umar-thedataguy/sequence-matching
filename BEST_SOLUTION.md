# Our final policy, stage by stage

**Team:** ML_Wizards_IITM · Round 2 · simulator release 1.0.0 · default mode `submission`

This document describes the policy we ship and why each choice was made. The experiment IDs in brackets (A1, B25, …) refer to [EXPERIMENTS_BY_STAGE.md](EXPERIMENTS_BY_STAGE.md), which lists every experiment, including the options we rejected. All results come from the public synthetic simulator and say nothing about real people.

## Results at a glance

- **Valid in every episode:** 120/120 through the organisers' harness, about 0.2 s per call (limit 10 s).
- **Better than greedy on every tie-breaker** (360 paired worlds, all significant): coverage +0.005, mutual acceptances +0.30 per 100, 8 fewer clarification units, first introduction 1.6 days sooner.
- **Primary score (MSMI):** level with greedy within noise (+0.019 ± 0.052 per 100), and +0.017 ± 0.011 qualifying pairs per episode in luck-free expected value (significant).
- **Close to the ceiling:** coverage 0.345 against a maximum possible 0.348.

## The policy in one picture

```
every day, two calls
┌─────────────────────────── A. ASKING (ask phase) ───────────────────────────┐
│ 1. score every askable person by Bayesian expected partners                 │
│    skip anyone under a 5% chance today; ask the top 4 bundles (12 units)    │
└──────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────── B. MATCHING (match phase) ───────────────────────┐
│ 2. feasible pairs only (organisers' eligibility, available, never introduced)│
│ 3. value of each pair = P(A says yes) × P(B says yes), 14-feature model      │
│ 4. exact blossom matching with tiered weights:                               │
│    most introductions ≫ most first-time people ≫ scarce people ≳ value       │
└──────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────── C. REPEAT (every call) ──────────────────────────┐
│ 5. no memory: rebuild everything from today's observation; never hold back  │
│ 6. update each person's reply and yes rates from visible feedback           │
│ 7. validate every ask and pair; fall back to greedy, then to empty lists    │
└──────────────────────────────────────────────────────────────────────────────┘
```

Configuration (`smp/pipeline.py`):

```python
'submission': dict(ask='bayes', match='tiered', score='logit', skip_below=0.05, scarcity=True)
```

---

## Stage A: Asking

### Part 1. Who to ask, and what

**What we do.**
1. Spend the budget only on **hard-constraint bundles** (3 units, all 11 deal-breakers at once), never on soft questions.
2. Only people who are available, have an unknown deal-breaker and have not declined one are askable.
3. Rank them by the **Bayesian expected number of partners**. We imagine each person's missing deal-breakers 12 times, each time borrowing a complete answer set from someone already observed and fitting it to their own age and zone, and count how many people they'd be compatible with.
4. Ask the top 4 today. Anyone whose chance of *any* partner is below 5% is skipped for today and re-scored tomorrow.

**Why.**
- Clarification is the largest lever in the problem: without it, coverage falls by 20.8 points and mutual acceptances by 7.3 per episode (A1).
- Soft questions improve prediction but not outcomes, and they cost 74 units (A4).
- The Bayesian score predicts who has a partner much better than counting possible partners: AUC 0.73 vs 0.59 (A9).
- In full episodes, the Bayesian order brings first introductions **1.0 day sooner** and adds **1.3 introductions per episode**. The skip saves **7.5 units per episode** with the same people served (A10).
- A higher skip threshold saves a little more but starts to lose coverage (A11).

**Code:** `smp/clarify.py`, `bayes_bundle_asks`, `match_chances`, `compatible`. Worked example: [ASK_STRATEGY.md](ASK_STRATEGY.md).

---

## Stage B: Matching

### Part 2. Who can meet

**What we do.** A pair is considered only if the organisers' `eligibility(a, b)` says `feasible`, both people are available, and they have never been introduced. No score can override this.

**Why.** One invalid pair invalidates the episode and makes the entry ineligible. Unknown deal-breakers always mean "not yet", never "probably fine" (B1).

**Code:** `smp/feasibility.py`, `feasible_edges`.

### Part 3. Scoring a pair

**What we do.**
- A logistic model predicts whether each person says yes to the other. It uses **14 features**, exactly the observable terms of the simulator's response equation:
  - agree / disagree / unknown on relationship goal, pace, lifestyle and conversations;
  - the number of unknown core fields;
  - the person's own reply rate and yes rate so far, with how much history there is;
  - how other people have answered this person.
- History is shrunk toward the training-pool rates (0.797 reply, 0.479 yes, strength 3), so a newcomer gets exactly the population rate.
- A pair's value is P(A yes) × P(B yes). It is used **only to rank** pairs, because the two answers are slightly correlated.

**How it was trained.**
1. Run 720 simulator episodes: greedy, random-feasible and a learned policy × 40 seeds × 6 families, each family in its own world.
2. Log the features at the moment of each decision, and the reply that came later: 110,700 rows.
3. Validate on held-out seeds: AUC 0.5717, calibrated.
4. Refit on all seeds and export to `model_weights.json` (2.8 KiB). The model is evaluated in plain Python inside the container.

**Why this model.**
- Only four soft fields matter (B5), unknown is a third state (B8), and the effects add up (B7).
- Extra features (age gap, zone, other soft fields, day) add nothing (B15, B16).
- Boosted trees, latent-trait and full probabilistic models are no better (B12).
- A second-meeting model is barely above chance (B17), and within-episode re-learning gives no gain (B18).
- The accuracy limit comes from the data, not the model: only 5.6% of each person's hidden generosity is learnable in time (B10).

**Code:** `smp/model.py`, trained by `train_shipped.py`.

### Part 4. Choosing today's pairs

**What we do.** Exact maximum-weight matching on the whole pool (Edmonds' blossom algorithm, standard library only), with integer weights in strict tiers:

```
w(pair) = 10^12                                   one introduction
        + 10^9  × number of first-time people     coverage, the first tie-breaker
        + 10^6  × scarcity of each first-time person   3, 2, 1 when they have 1, 2, 3 feasible partners left
        + round(10^6 × predicted pair value) + 1  remaining choices
```

The matcher therefore makes **as many introductions as possible**. Among those, it serves **as many new people as possible**. Among those, it protects people who are about to run out of partners. Only then does it prefer the most promising pairs.

**Why.**
- An unused feasible pair is usually lost for good, and perfect knowledge of pair values moves MSMI by only ~4–6% (B22).
- The feasible graph is not bipartite, and greedy loses value (the problem statement's A–B / C–D example). The blossom matcher is exact on 3,000 test graphs (B24).
- The coverage tier raises coverage, proven, at no measurable cost to MSMI (B25).
- The value term protects mutual acceptances (B25, no-value ablation).
- The scarcity bonus adds 50 served people over 360 episodes and shortens waits (B26).

**Code:** `smp/allocate.py`, `tiered_weights`; `smp/matching.py`, `max_weight_matching`, `best_pairs`.

---

## Stage C: Repeat

### Part 5. The daily re-run

**What we do.** The policy keeps no memory and returns `{}` unchanged. Every call rebuilds its view from the observation. This covers new arrivals (up to day 20), people busy for 8 days after an introduction, people who left without notice, and people paused after a success. Nobody who has a feasible partner today is held back.

**Why.**
- Departures are random and invisible to the policy (C2).
- Acceptance falls after day 35 in the drift family (C4).
- Waiting for better partners showed no measurable gain (C1).
- Tested directly, waiting **hurts**. Skipping pairs whose predicted value is in the bottom quarter raises the quality of each introduction (mutual yeses per introduction +0.014 ✱) but loses 11 introductions per episode, so MSMI falls by 0.042 per 100 ✱ and coverage by 0.018 ✱. Requiring known soft agreements is far worse: coverage falls from 0.344 to 0.148, because 72% of introduced pairs have no known agreement. See "Waiting for a better match" in [EXPERIMENTS_BY_STAGE.md](EXPERIMENTS_BY_STAGE.md#waiting-for-a-better-match).
- Being memory-free also makes every decision reproducible from the request alone and rules out leakage between episodes.

### Part 6. Learning from feedback

**What we do.** Reply and yes rates are recomputed each call from the feedback already visible. Silence lowers the reply rate, never the willingness to accept. Introductions still awaiting replies are not counted as failures.

**Why.** Person effects exist (B9) but little can be learned in an episode (B10). Exploration and re-learning gave nothing (C6, C7).

### Part 7. Never invalid, always fast

**What we do.**
1. Records that break the data contract are set aside, never repaired.
2. Every ask and pair is re-checked against the simulator's own rules, and anything invalid is dropped.
3. If our code fails, the organisers' greedy rule is used, and as a last resort empty lists, which are always valid.
4. A 6 s compute budget (the limit is 10 s per call) switches to a greedy matcher if fewer than 2 s remain.

**Why.** One invalid assessed episode makes the whole entry ineligible. There are 42 tests, including 120 corrupted observations and full episodes needing zero interventions. The slowest call measured is 0.31 s (C8–C10).

**Code:** `smp/safety.py`, `smp/normalise.py`, `policy.py`.

---

## What the final policy achieves

`python experiments.py --configs greedy,no_asks,random,final,tiered,submission --seeds 6001-6060 --independent`

The run covers 360 independent worlds (60 fresh seeds × 6 families) and 2,160 episodes. Every policy runs on the same worlds, so all differences are paired. The three organiser baselines are run exactly as released.

### Mean per episode, in the organisers' units

| Policy | **MSMI per 100** (primary score) | **Coverage** (1st tie-breaker) | **Mutual acceptances per 100** (2nd) | **Clarification cost** (3rd, units) | Introductions | Dates | Wait to first introduction (days) |
|---|---|---|---|---|---|---|---|
| Greedy (organisers) | 0.390 | 0.338 | 5.42 | 200.5 | 75.6 | 8.6 | 6.90 |
| No clarification (organisers) | 0.107 | 0.120 | 1.50 | 0.0 | 21.0 | 2.4 | 3.91 |
| Random feasible (organisers) | 0.406 | 0.339 | 5.65 | 200.5 | 75.6 | 8.9 | 6.91 |
| Round 1 policy (`final`) | 0.404 | 0.340 | 5.59 | 200.5 | 75.1 | 8.8 | 6.70 |
| Tiered matcher only (`tiered`) | 0.422 | 0.343 | 5.64 | 200.5 | 75.3 | 8.8 | 6.42 |
| **Shipped policy (`submission`)** | **0.410** | **0.343** | **5.72** | **192.5** | **76.3** | **9.0** | **5.34** |

### Shipped policy minus each baseline (paired, 95% interval over 60 seeds; ✱ = proven)

| Metric | vs Greedy | vs Random feasible | vs No clarification |
|---|---|---|---|
| MSMI per 100 | +0.019 ± 0.052 | +0.004 ± 0.051 | **+0.303 ± 0.043 ✱** |
| Coverage | **+0.0051 ± 0.0008 ✱** | **+0.0039 ± 0.0007 ✱** | **+0.224 ± 0.004 ✱** |
| Mutual acceptances per 100 | **+0.30 ± 0.18 ✱** | +0.08 ± 0.18 | **+4.22 ± 0.15 ✱** |
| Clarification cost (units) | **−8.0 ± 0.4 ✱** | **−8.0 ± 0.4 ✱** | +192.5 ± 2.0 (no clarification asks nothing) |
| Introductions | **+0.65 ± 0.55 ✱** | **+0.65 ± 0.58 ✱** | **+55.3 ± 1.3 ✱** |
| Dates | **+0.41 ± 0.32 ✱** | +0.10 ± 0.30 | **+6.6 ± 0.3 ✱** |
| Wait to first introduction (days) | **−1.56 ± 0.10 ✱** | **−1.57 ± 0.11 ✱** | +1.43 ± 0.20 (it introduces far fewer people) |

### By family (MSMI per 100 / coverage)

| Family | Greedy | No clarification | Random feasible | **Shipped** |
|---|---|---|---|---|
| development | 0.467 / 0.385 | 0.133 / 0.145 | 0.408 / 0.387 | **0.550 / 0.392** |
| sparse | 0.117 / 0.162 | 0.025 / 0.047 | 0.092 / 0.163 | 0.092 / 0.163 |
| cold_start | 0.417 / 0.329 | 0.042 / 0.064 | 0.417 / 0.330 | 0.408 / 0.334 |
| delayed | 0.442 / 0.384 | 0.208 / 0.156 | 0.533 / 0.385 | 0.500 / 0.391 |
| shift | 0.433 / 0.381 | 0.117 / 0.153 | 0.450 / 0.383 | 0.492 / 0.386 |
| drift | 0.467 / 0.387 | 0.117 / 0.151 | 0.533 / 0.389 | 0.417 / 0.392 |

Per-family MSMI rests on 20–35 qualifying pairs per cell, so differences of ±0.1 are within chance. Coverage is higher than greedy's in every family.

### Reading

- **Ranking metrics.** Against the organisers' greedy reference, the shipped policy is proven better on the first tie-breaker (coverage), the second (mutual acceptances) and the third (clarification cost, 8 units less).
- **Outcome funnel.** It also makes more introductions and gets more dates, and introduces people **1.6 days sooner**.
- **Primary score.** MSMI per 100 is higher (0.410 vs 0.390), but the difference (+0.019 ± 0.052) cannot be separated from chance with 360 episodes. This matches the head-room analysis: no policy can move MSMI by much more than a few percent.
- **Clarification.** Switching it off is by far the worst choice (coverage 0.12, MSMI 0.107), which is the clarification ablation (A1).
- **Run time.** The slowest single call in this run was 1.7 s while three studies shared the CPU (limit 10 s). Inference time measured through the organisers' own harness, one process per call, is reported below.

### Through the organisers' own harness: every metric, including inference time

`python run_final_study.py --seeds 101-120 --output-dir results/official`

This is exactly the organisers' protocol. It runs `evaluate.episode` with **a new policy process for every call**, JSON over stdin and stdout, and our policy started with no arguments, as in the container. The organisers' seed layout is used: seeds 101–120 in each of the 6 families, 120 episodes per policy, 600 in total. The per-policy files are in `results/official/*.json`, in `evaluate.py` format.

| Organiser metric | Greedy | No clarification | Random feasible | Round 1 policy | **Shipped** |
|---|---|---|---|---|---|
| Valid episodes | 120 / 120 | 120 / 120 | 120 / 120 | 120 / 120 | **120 / 120** |
| **Primary score**: MSMI per 100 arrived members | 0.433 | 0.183 | 0.367 | 0.358 | **0.354** |
| Qualifying pairs, total over 120 episodes | 104 | 44 | 88 | 86 | **85** |
| **Coverage** | 0.3448 | 0.1230 | 0.3447 | 0.3460 | **0.3490** |
| **Mutual acceptances per 100** | 5.19 | 1.73 | 5.52 | 5.54 | **5.51** |
| **Clarification cost** (units per episode) | 200.4 | 0.0 | 200.4 | 200.4 | **193.0** |
| **Inference time** (seconds per episode, 120 calls) | 27.2 | 28.3 | 26.2 | 21.1 | **23.5** |
| Inference time per call (seconds, including start-up) | 0.227 | 0.236 | 0.218 | 0.176 | **0.196** |
| Assignments (introductions) | 75.4 | 20.9 | 75.3 | 75.3 | **77.6** |
| Mutual acceptances | 10.38 | 3.46 | 11.04 | 11.07 | **11.03** |
| Dates | 8.32 | 2.68 | 8.55 | 8.88 | **8.47** |
| Missing feedback (non-responses) | 38.9 | 10.5 | 39.1 | 39.6 | **41.6** |
| Unserved members (of 200) | 131.0 | 175.4 | 131.1 | 130.8 | **130.2** |
| First-introduction wait, mean / median (days) | 6.75 / 5 | 3.82 / 0 | 6.58 / 5 | 6.45 / 5 | **5.23 / 3** |

**Shipped minus greedy, paired over the 20 seeds** (✱ = proven):

| Metric | Difference |
|---|---|
| Coverage | **+0.0042 ± 0.0024 ✱** |
| Unserved members | **−0.83 ± 0.48 ✱** |
| Clarification cost | **−7.4 ± 1.0 ✱** |
| Inference time per episode | **−3.7 ± 0.4 s ✱** |
| Introductions | **+2.2 ± 1.9 ✱** |
| Missing feedback | +2.7 ± 2.2 ✱ (a count: more introductions give more non-responses) |
| Mutual acceptances per 100 | +0.32 ± 0.49 |
| MSMI per 100 | −0.079 ± 0.154 (not distinguishable from zero) |

**Reading the primary score honestly.** On these 120 episodes greedy recorded 104 qualifying pairs and the shipped policy 85. Totals of this size vary by about ±14 from chance alone, and the paired interval (−0.079 ± 0.154) is wide. This run also holds less evidence than it looks: the organisers' layout repeats each seed across families, and four families share the same people within a seed.

The three larger tests agree that the shipped policy is **not worse** on MSMI:
- the 360-world paired study above: +0.019 ± 0.052;
- the luck-free expected MSMI on 180 worlds: **+0.017 ± 0.011 ✱** per episode (head-room section below);
- the per-component tests in [EXPERIMENTS_BY_STAGE.md](EXPERIMENTS_BY_STAGE.md).

The organisers rank on 120 private episodes, so the primary score will carry the same luck for every team. The tie-breakers, where the shipped policy is consistently and provably ahead (coverage, clarification cost, inference time), are what we can control.

**By scenario family, through the organisers' harness.** Each cell shows MSMI per 100 / coverage / clarification cost / mean wait to first introduction (days) / missing feedback / inference seconds per episode, averaged over 20 episodes per cell.

| Family | Greedy | No clarification | Random feasible | **Shipped** |
|---|---|---|---|---|
| development | 0.675 / 0.391 / 192 / 6.3 / 45.9 / 28.4 | 0.325 / 0.155 / 0 / 3.8 / 13.8 / 27.7 | 0.500 / 0.391 / 192 / 6.0 / 46.6 / 26.3 | 0.450 / **0.395** / **189** / **4.9** / 50.1 / **24.0** |
| sparse | 0.075 / 0.171 / 196 / 6.8 / 11.4 / 29.6 | 0.000 / 0.052 / 0 / 3.1 / 2.5 / 29.8 | 0.075 / 0.171 / 196 / 6.9 / 11.5 / 26.4 | 0.075 / 0.171 / **172** / **4.9** / 11.2 / **22.8** |
| cold_start | 0.300 / 0.336 / 237 / 8.7 / 38.6 / 26.6 | 0.100 / 0.066 / 0 / 3.8 / 5.5 / 29.5 | 0.275 / 0.335 / 237 / 8.7 / 38.8 / 25.9 | **0.450** / **0.341** / **231** / **6.7** / 39.9 / **22.6** |
| delayed | 0.375 / 0.390 / 192 / 6.5 / 45.9 / 24.9 | 0.150 / 0.155 / 0 / 4.1 / 13.7 / 29.3 | 0.375 / 0.391 / 192 / 6.3 / 46.4 / 25.4 | 0.325 / **0.396** / **189** / **5.2** / 48.3 / **23.6** |
| shift | 0.550 / 0.391 / 192 / 6.3 / 45.7 / 26.0 | 0.200 / 0.155 / 0 / 3.8 / 14.1 / 26.3 | 0.525 / 0.389 / 192 / 6.0 / 45.0 / 25.3 | 0.375 / **0.396** / **189** / **4.9** / 49.9 / **24.0** |
| drift | 0.625 / 0.391 / 192 / 6.3 / 46.1 / 27.9 | 0.325 / 0.155 / 0 / 3.8 / 13.8 / 27.2 | 0.450 / 0.391 / 192 / 6.0 / 46.5 / 27.6 | 0.450 / **0.395** / **189** / **4.9** / 50.2 / **23.8** |

Each MSMI cell rests on 0–14 qualifying pairs, so per-family MSMI differences of this size are within chance. Coverage, clarification cost, waiting time and inference time are at least as good as greedy's in every family.

**Inference time.** The figures were measured with 10 episodes running in parallel on a 12-thread laptop, so they are conservative. The shipped policy is faster than the baselines because it scores only complete members, while the reference matcher checks every available pair. The limit is 10 s per call; the slowest episode averaged 0.22 s per call.

## How close this is to the maximum

`python headroom_r2.py --seeds 4001-4030` is a **diagnostic**: it reads the simulator's hidden values and is never part of the policy. It covers 180 worlds (30 seeds × 6 families, independent), with greedy and our policy run on the same worlds. Our policy here is the version before the scarcity bonus (`submission_v1`); the bonus adds a further +0.0007 coverage (B26).

**How the maximums are computed**

- **Coverage ceiling.** The share of arrived members who have at least one *usable* partner: neither has declined a deal-breaker, their true answers are compatible in both directions, and both are in the pool on a common decision day. No policy can serve anyone else.
- **Expected MSMI of an introduction.** The exact probability that it qualifies, from the simulator's own formula:
  - both reply × both say yes;
  - × 0.78 that the date happens;
  - × P(date within 30 days);
  - × each person replies, says yes to a second meeting and answers within 3 days.

  This is averaged over the pair's shared random term. Summed over a policy's introductions, it gives that policy's MSMI with the luck removed.
- **MSMI upper bound (slot-capped).** A linear programme chooses pairs to maximise expected MSMI when each member can take at most one introduction per 8 days they are in the pool. It is unreachable, because it ignores that both people must be free on the same day.

**Results, in the organisers' units** (per episode of 200 members; ± = 95% interval of the paired difference over 30 seeds):

| Metric | Maximum possible | Greedy (reference) | **Ours** | Ours − greedy |
|---|---|---|---|---|
| Coverage (share of 200 members introduced at least once) | 0.348 | 0.339 | **0.345** | **+0.0054 ± 0.0012 ✱** |
| Members short of the coverage ceiling | 0 | 1.72 | **0.64** | 1.1 more members served per episode |
| Expected MSMI, qualifying pairs per episode (luck removed) | ≤ 0.972 | 0.777 (80.0% of bound) | **0.794 (81.7%)** | **+0.017 ± 0.011 ✱** |
| Expected MSMI per 100 arrived members | ≤ 0.486 | 0.389 | **0.397** | +0.008 |
| Realised MSMI per episode (observed count) | | 0.73 | 0.84 | +0.11 ± 0.17 (luck dominates) |

**By family:**

| Family | Coverage ceiling | Greedy | Ours | MSMI bound | Greedy expected MSMI | Ours expected MSMI |
|---|---|---|---|---|---|---|
| development | 0.394 | 0.382 | 0.391 | 1.130 | 0.933 | 0.940 |
| sparse | 0.174 | 0.171 | 0.173 | 0.250 | 0.239 | 0.240 |
| cold_start | 0.331 | 0.321 | 0.328 | 1.026 | 0.818 | 0.823 |
| delayed | 0.400 | 0.389 | 0.394 | 1.232 | 0.947 | 0.956 |
| shift | 0.408 | 0.400 | 0.404 | 1.078 | 0.862 | 0.888 |
| drift | 0.380 | 0.371 | 0.377 | 1.115 | 0.864 | 0.915 |

**Reading:**
- **Coverage.** We close about two-thirds of greedy's gap to the ceiling, in every family.
- **MSMI.** Our introductions are worth a proven +2% more qualifying pairs than greedy's, at 82% of an upper bound that no policy can reach. The realised MSMI count cannot show a difference this small, because a typical episode has only 0–2 qualifying pairs.
- **Sparse geography.** It is supply-bound: both policies are within 0.003 of the ceiling.

---

## Hard scenarios and failure cases

| Scenario | What goes wrong | What the policy does | Evidence |
|---|---|---|---|
| **Sparse geography** | Only ~23 truly feasible pairs per episode; the coverage ceiling is 0.174 | Every feasible pair is used; the Bayesian skip saves the most here (172 vs 196 units) | Within 0.001 of the coverage ceiling (head-room table) |
| **Incompatible candidates** | ~65% of members have no truly compatible partner at all (35% declined a deal-breaker) | They are reported as unserved and never forced into a pair; declined members are never asked | Unserved 130.2 of 200 in the official run; on the head-room worlds about 130 of 200 can never be served by any policy |
| **Withheld constraints** | A declined deal-breaker can never become known | The member is never asked and never introduced: eligibility treats unknown as "not yet" | Contract tests; part 2 |
| **Cold start** | Only 20% of profiles start complete; there is no history | Bayesian ask order unlocks likely matches first (wait −1.6 days in cold start); population priors for newcomers | A10 by family |
| **Delayed dates** | Dates 5–12 days later; some miss the 30-day window | Introduce early; never hold back | MSMI window computed exactly in the head-room analysis |
| **Shifted outcome weights** | The soft fields matter differently (lifestyle agreement lowers acceptance) | The value term only orders choices after cardinality and coverage, so mis-ranking costs little | Model AUC 0.548 in shift vs 0.557–0.588 elsewhere (B16); re-learning gave no gain (B18) |
| **Changing response conditions (drift)** | Acceptance falls after day 35 | Front-loading; a uniform drop does not change which pair is best on a day | C4 |
| **Competition for the same candidate** | 57% of unused feasible pairs are lost this way | Exact global matching with coverage and scarcity tiers | B22, B25, B26 |
| **Low event counts** | 0–2 qualifying pairs per episode; MSMI comparisons are noisy | We report paired intervals over seeds and the luck-free expected MSMI | Head-room section |

## Limitations

- **MSMI gains are small and hard to prove.** Even perfect knowledge of every pair moves MSMI by only ~4–6% over greedy. Our proven gains are in coverage, mutual acceptances, clarification cost, waiting time and inference time. Our MSMI is equal to greedy's within the noise, and +2% in luck-free expected value.
- **The yes-model learns this simulator's rules.** Held-out accuracy is measured on new worlds with the same rules. When the rules change (the shift family), accuracy drops to 0.548, and nothing we tried recovers it.
- **The pair value is not a calibrated joint probability.** The two directions are positively correlated (ρ = 0.037), so the product is used only to rank.
- **The Bayesian ask model is over-confident at the extremes** (13% of people scored below 5% did have a partner). Re-scoring daily makes the skip safe in our tests, but private worlds with other rules could differ.
- **Selective labels.** Outcomes exist only for introduced pairs and logged propensities are null, so we make no claim of unbiased off-policy evaluation. Every comparison is a complete simulator episode on paired seeds.

## What these results cannot establish, and what real use would need

Every person, answer and outcome here is synthetic. The simulator's response formula is invented, so the learned weights, probabilities and scores say **nothing** about real people or real relationships, and must not be presented as compatibility estimates.

What carries over is the decision machinery:
- deal-breakers are never overridden;
- missing information is asked for before it is assumed;
- unknown is kept separate from "no";
- the whole pool is allocated at once instead of pair by pair;
- no action is ever emitted unvalidated.

Authorised later use (see `docs/INTEGRATION.md`) would need:
1. an adapter that maps authorised observations into this schema without inventing defaults for missing fields;
2. a fresh check of every constraint against current data at decision time;
3. retraining and calibration on authorised, consented data, with fairness checks across groups, before any probability is trusted;
4. human review of every suggested introduction, and an explanation that lists the observed evidence and the unresolved information behind it.
