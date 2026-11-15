# Fraud Ring Detector 🕵️‍♂️
### Can a graph neural network catch fraud that a plain tree model misses? I tested it properly. Here's what happened.

## TL;DR
- I built a pipeline (Kedro + PyTorch Geometric) that scores every transaction for fraud and compares three models, simple to fancy.
- The biggest win came from **relationship features**, like "how many customers share this card?". They lifted a plain tree model
  from **0.13 to 0.73 PR-AUC** on my banking-style data.
- My neural network first looked *worse* than the tree model. Then I read the training log: it had made only 60 weight updates.
  Trained longer, it matched the tree model (**0.729 vs 0.725**). My first conclusion was wrong, and I fixed it.
- I tried to find laundering **rings**. I didn't find any, and I tell you exactly why below.
- Everything runs on synthetic data, so none of this is a claim about real banks.

## Why I built this (and the embarrassing part)
I wanted to know one thing: can a graph neural network spot fraud that ordinary tabular models miss, and in particular, can it
find the *rings*, the groups of accounts that move money together?

So I built a Kedro pipeline that scores every transaction by combining:

- GraphSAGE embeddings of the accounts involved (customers, merchants and, where the data has them, shared cards and devices),
- a 1D-CNN over the sender's recent history, and
- the transaction's own features.

Then it looks at the flagged transactions as a directed network (communities, overlap, roles, cycles) to surface **candidate**
syndicates. And everything is compared against a ladder of gradient-boosting baselines, because a graph model that can't beat a
well-fed tree model hasn't earned its complexity.

Here's the embarrassing part. **Version 1 of this project looked great and learned nothing.** The features were random numbers,
the labels were random, and the AUC was 0.5 by construction. An audit humbled me, and the whole repo got rebuilt around one rule:
**never trust a number I haven't tried to break.** That means a time-based split, leakage checks, PR-AUC instead of ROC-AUC,
honest baselines, bootstrap intervals, and a profiler that tells me when a dataset can't support my claim.

> This was inspired by the GCN+CNN idea in Cai (2026), *Int. J. Comput. Intell. Syst.* 19:186. It's an **extension, not a
> replication**. See "How this differs from the paper" below.

## The idea in 30 seconds
Picture transactions as dots and lines: customers, merchants, cards and devices are the dots, and "paid" or "used" are the lines.
Five customers sharing one card looks harmless one payment at a time. In the picture, it's a red flag.

I compare three models on the same data, so I can see what each ingredient actually adds:

| | Model | In plain words |
|---|---|---|
| **M1** | Gradient boosting on the payment's own features | The simple baseline |
| **M2** | M1 + hand-made relationship and history counts | "Give the simple model the right clues" |
| **M3** | Hybrid GraphSAGE + 1D-CNN | The fancy one that reads the graph itself |

If M3 can't beat M2, the graph network hasn't proved anything the counts couldn't already do.

## Two tracks
| | **Track B: fraud scoring (run, real data)** | **Track A: ring detection (three real windows run)** |
|---|---|---|
| Question | What does graph structure add beyond strong tabular features? | Can graph models recover named laundering patterns? |
| Data | Mendeley-style banking file: customers, merchants, cards, devices | IBM AMLworld (HI-Small): account-to-account transfers + a pattern file |
| Fraud types | Card testing, geo anomalies, account takeover, a few laundering rings | Fan-in/out, cycles, scatter-gather, gather-scatter, bipartite, stack, random |
| Evidence | Baseline ladder, recall per fraud type | Same ladder + pattern-level recall + bootstrap intervals |
| Run with | `kedro run --env mendeley` | `kedro run --env amlworld` |

## What Track B showed (v0.4.1, Mendeley-style data)
The setup: 310,906 transactions (about 31% of the non-fraud rows, keeping every fraud customer's full history), a strict
time-held-out test period, and the sampling correction applied (real fraud rate: 0.149%).

| Model | PR-AUC (95% interval) | Recall | Precision in the real population* | Alerts per 10,000 |
|---|---|---|---|---|
| M1 GradBoost, tabular only | 0.128 (0.097 to 0.161) | 0.44 | 4.1% | 162 |
| M2 GradBoost + graph/sequence features | **0.725 (0.668 to 0.784)** | 0.72 | 93.0% | 11.5 |
| M3 Hybrid GraphSAGE+CNN (+card/device nodes) | 0.646 (0.590 to 0.705) | 0.58 | 62.7% | 13.8 |

\*Corrected for the sampling; approximate, because the sample keeps all fraud customers' histories.

Quick reality check on what those last two columns mean: M1 makes an analyst look at about 162 alerts per 10,000 transactions and
only 4% of them are real. M2 makes them look at about 12, and 93% are real. That's the difference between a smoke alarm that rings
for burnt toast and one you can trust.

The paired bootstrap over sender accounts (200 resamples) says **M2 minus M1 = +0.60 (0.54 to 0.65)** and **M3 minus M2 = -0.081
(-0.118 to -0.052)**, with none of the 200 resamples favouring M3 at that point. That interval only measures test-sample wobble,
not training variability or a different time window. An earlier 3-seed run (model initialisation only) gave 0.124 / 0.723 / 0.656.

Recall by fraud type (test period; *n* = transactions):

| Fraud type | n | M1 | M2 | M3 |
|---|---|---|---|---|
| card_testing | 143 | 0.72 | 1.00 | 1.00 |
| geo_anomaly | 76 | 0.01 | 0.00 | 0.00 |
| account_takeover | 40 | 0.00 | 0.95 | 0.00 |
| money_laundering_ring | 19 | 1.00 | 0.95 | 1.00 |

### So what does it all mean?
1. **The signal is shared infrastructure.** Take card and device information away and M2 falls from 0.725 to 0.201 PR-AUC;
   account takeover recall goes from 0.95 to 0 and card testing from 1.00 to 0.45. Almost all of M2's gain is relational
   (how many customers share a card or device, how many devices a customer has used), computed by hand from the graph.
2. **Plot twist: my first conclusion about the neural network was wrong.** With the default budget (60 full-batch epochs, which
   means only 60 weight updates) M3 scored 0.646 and never learned account takeover. I almost wrote that up as "the network can't
   do it." Then I trained the same network for 300 epochs with a larger step and no neighbour messages, and it scored
   **0.729 (0.671 to 0.785) against M2's 0.725 (0.668 to 0.784)**. The gap to M2 is +0.003 (0.001 to 0.007), and account takeover
   recall went from 0.00 to 0.95. The network wasn't unsuited to the problem. It was undertrained.
3. **Message passing isn't what closed the gap.** The relational information already sits in the engineered features that both
   models receive. Whether learned message passing adds anything *on top* once training is adequate is still untested
   (`kedro run --env longtrain`, which is slow). At the default budget, GraphSAGE (0.646) and no messages (0.656) differed by
   about the size of run-to-run noise (two identical runs differed by 0.009).
4. **At equal alert volume** (default training), the learned model matched M2 only at the very top of the ranking: at 5 alerts per
   10,000 M2 and M3 catch 34% and 32% of fraud, at 20 per 10,000 it's 72% and 63%, and the tabular-only model catches 13%.
   With adequate training, the long-trained network's recall matches M2 at every volume I looked at.
5. **Nobody catches geo anomalies**, which are 27% of the test positives. That's an error-analysis job before it's a modelling job.
6. **Rings are only 19 test transactions here**, and even the tabular-only model catches them all, so ring claims from this
   dataset would be anecdotal. That's exactly why Track A exists.
7. **A leak I almost believed.** An early profiler run said a plain tabular model scored 0.965 PR-AUC. I now think that was a leak
   (`transaction_id` was being used as a feature). The honest tabular score is the 0.128 above. When a number looks too good, hunt
   for the leak first.

**In plain words:** graph information helps a lot, in the form of relational features. Once the network is trained properly it
matches the gradient-boosting model but doesn't beat it, and I haven't shown that message passing adds anything beyond the
features. I'd rather tell you that than claim a win for the graph network, and I only got here because I checked how the model was trained.

### The ablations and the training-budget test
Each one is its own environment and output folder:
```bash
kedro run --env ablation_no_entities    # no card/device nodes and no shared-card/device counts
kedro run --env ablation_no_graph       # M3's inputs, but no neighbour messages (60 epochs)
kedro run --env longtrain_no_graph      # same, 300 epochs, larger step: is the network just undertrained? (about 15-20 minutes)
kedro run --env longtrain               # with GraphSAGE messages, 150 epochs (slow, about 30 minutes)
```
| Variant | M2 PR-AUC | M3 PR-AUC | Account takeover recall (M2 / M3) | Card testing recall (M2 / M3) |
|---|---|---|---|---|
| Full, default training (`--env mendeley`) | 0.725 (0.668 to 0.784) | 0.646 (0.590 to 0.705) | 0.95 / 0.00 | 1.00 / 1.00 |
| No entities | 0.201 (0.156 to 0.245) | 0.111 (0.072 to 0.150) | 0.00 / 0.00 | 0.45 / 0.01 |
| No message passing, default training | 0.725 (same model) | 0.656 (0.601 to 0.714) | 0.95 / 0.00 | 1.00 / 1.00 |
| **No message passing, long training** | 0.725 (same model) | **0.729 (0.671 to 0.785)** | 0.95 / **0.95** | 1.00 / 1.00 |

M1 (tabular only) is identical across all runs (0.128), which is a handy consistency check. All intervals are bootstrap intervals
over sender accounts, on one trained model and one time window.

**What I'd do next:** run `kedro run --env longtrain` (does message passing add anything once training is adequate?) and
`python tools/run_all.py ... --aml-env amlworld_longtrain` (do the Track A conclusions about M3 survive adequate training?); look at
the geo-anomaly cases; try a stacked model (gradient boosting plus a learned score as a feature, built without leakage); and
check stability on a different time split.

## What Track A showed (IBM AMLworld HI-Small, three valid windows)
AMLworld is huge (about 5 million rows) and my laptop has 8 GB of RAM, so I used **time windows** of about 300,000 to 320,000
transactions, each with a chronological 60/20/20 split inside it. (Random row sampling would thin every account's history and wreck
the graph, so no.)

| Window | Placed by | Hours | Transactions | Laundering rows | Laundering in the test period |
|---|---|---|---|---|---|
| time 0.3 | clock | 127 to 143 | 311,621 | 395 (0.13%) | 42 |
| time 0.5 | clock | 212 to 239 | 306,128 | 504 (0.16%) | 132 |
| rows 0.85 | transaction count | 195 to 207 | 320,099 | 267 (0.08%) | 47 |

A fourth window (time 0.7) was **degenerate, so I excluded it**: it held 374 transactions, 59% of them laundering, because the
file's activity thins out after about hour 297. Every model scored a perfect 1.000 there, which is exactly the kind of result that
should make you suspicious. The tools now refuse such windows. Also: about 40% of laundering rows aren't listed in the patterns file.
I checked that this isn't a matching bug on my side; the dataset itself leaves them unlisted.

PR-AUC with 95% bootstrap intervals over sender accounts:

| Window | M1 tabular | M2 + graph/flow features | M3 hybrid GraphSAGE+CNN | M3 minus M2 |
|---|---|---|---|---|
| time 0.3 | 0.016 (0.008 to 0.029) | **0.133 (0.061 to 0.294)** | 0.052 (0.021 to 0.140) | -0.088 (-0.178 to -0.003) |
| time 0.5 | 0.045 (0.032 to 0.059) | **0.151 (0.095 to 0.220)** | 0.036 (0.019 to 0.065) | -0.114 (-0.177 to -0.064) |
| rows 0.85 | 0.034 (0.013 to 0.114) | **0.070 (0.030 to 0.167)** | 0.001 (0.001 to 0.001)\* | -0.079 (-0.166 to -0.029) |

\*M3 did not train in this window: its best validation score came at epoch 1, so training stopped at epoch 11, and its test ROC-AUC is 0.44.

To compare models fairly I look at recall at **equal alert volume**: take the top-ranked transactions up to 20 or 50 alerts per
10,000, so every model is judged at the same operating point.

| Window | Recall at 20 (M1 / M2 / M3) | Recall at 50 (M1 / M2 / M3) | Real share of alerts at 20 (M1 / M2 / M3) |
|---|---|---|---|
| time 0.3 | 0.07 / **0.36** / 0.17 | 0.14 / **0.45** / 0.24 | 5% / **23%** / 11% |
| time 0.5 | 0.06 / **0.24** / 0.08 | 0.17 / **0.31** / 0.18 | 5% / **20%** / 6% |
| rows 0.85 | 0.11 / **0.19** / 0.00 | 0.21 / **0.28** / 0.00 | 5% / **8%** / 0% |

### What I take from it
1. **Hand-built relational features win again.** M2 beats M1 and M3 at equal volume in every window; on average it finds 3.3 times as
   much laundering as the tabular-only model at 20 alerts per 10,000 (26% against 8%). Its PR-AUC interval excludes zero against M1
   in two of three windows.
2. **Read the M3 results with a caveat, and now I know it matters.** I train M3 full-batch, so 60 epochs is only 60 weight updates,
   and in rows 0.85 it failed to train at all. On the Track B data the same kind of undertraining turned out to explain M3's whole
   deficit, so I treat M3's Track A losses as results *at a small training budget*, not as evidence about learned graph models.
   `conf/amlworld_longtrain` re-tests it (`--aml-env amlworld_longtrain`).
3. **Absolute performance is low.** At 20 alerts per 10,000, about one alert in five is real in the better windows and about one in twelve in the third.
4. **Unlisted laundering is the blind spot.** At 20 alerts per 10,000 M2 catches 22%, 9% and 11% of it; M1 and M3 catch 6% or less.
5. **Pattern-level evidence is anecdotal.** The test periods hold 1, 16 and 2 listed pattern instances, and I decided in advance
   that anything under five instances counts as anecdotal.
6. **My syndicate layer found no rings.** Built from M2's flags (23 to 35% of the flagged transactions are laundering), the flagged
   transactions were isolated pairs: no group of three or more accounts and no cycles in any window. At these alert volumes the
   flagged edges are too sparse to link up, so ring *detection* is not supported by my results.

I wrote my decision rules down before the first run (`docs/TRACK_A_PLAN.md`). The outcome: M2 beat M3 with an interval below zero in
all three windows (rule 2, so the headline is that hand-built relational features win, qualified by point 2); every pattern type is
anecdotal (rule 3); I have three valid windows, so rule 4 is met; and I make no claim of detecting syndicates (rule 5).

## How this differs from the paper that inspired it
Cai (2026) combines a GCN over accounts with a CNN over transaction features and reports very strong results. I took the idea
as inspiration, but this is **not a replication**:

- **Different parts:** GraphSAGE instead of GCN (it can score accounts it hasn't seen), and a CNN over each sender's real last-10 history instead of one transaction's reshaped features.
- **Different data:** PaySim looked too sparse for graphs, so I used a banking-style dataset and IBM AMLworld.
- **Stricter testing:** a split by time, PR-AUC and equal-alert-volume recall instead of accuracy (fraud is rare, so accuracy flatters), bootstrap intervals, ablations and a training-budget test.
- **Different baselines:** a ladder I built myself, including a strong tree model with hand-made relationship counts.
- **Different conclusions:** large gains from relationship information, but no evidence that learned message passing adds more, and no rings found.
- **Numbers aren't comparable:** the paper's headline figures were measured on different data with a different split and different metrics. Some of its figures also don't agree with each other, so I treated it as a source of ideas, not a benchmark to beat. I'm not saying it's wrong about its own setup; I'm saying I couldn't reproduce or test its claims.

## Status and limitations (the honest list)
- Track B has run on real data: a 3-seed run, a bootstrap run and the ablations, all on one sample and one time split. Track A has run on three valid AMLworld windows; a fourth was excluded as degenerate.
- Synthetic datasets reflect their generators, not real banks. Every number here is "performance on this benchmark".
- A ring shows up as a cycle only if every hop is flagged, so ring recovery is bounded by model recall.
- Row sampling thins each account's history; I prefer `--customer-frac` (Track B) and a time window (Track A).
- AMLworld amounts are in different currencies and I have no exchange rates, so amounts are only roughly comparable.

---

# Technical reference

## Pipeline (Kedro DAG)
| Node | What it does |
|---|---|
| `prepare_dataset_node` | Keeps all fraud rows, caps legit rows, sorts by time, **time-based** train/val/test split, shared account index |
| `build_features_node` | Tabular features (incl. auto-detected extras), causal shared-entity counts, optional account-flow features, per-split graph snapshots, last-K sequence tensor |
| `train_baselines_node` | M1 (tabular) and M2 (tabular + graph/sequence features) gradient boosting |
| `train_hybrid_model_node` | Full-batch GraphSAGE (bidirectional edges) + 1D-CNN, focal loss, early stopping on validation PR-AUC |
| `evaluation_node` | PR-AUC, ROC-AUC, P/R/F1, population-adjusted precision and alerts/10k, recall per fraud type, bootstrap intervals, threshold sweep |
| `pattern_evaluation_node` | Track A: recall per laundering pattern type and per pattern instance (detected-any / detected-majority) |
| `syndicate_analysis_node` | Directed, amount-aware communities, overlap, roles, cycles on flagged transactions; table + PyVis dashboard |
| `neo4j_sync_node` | Optional: writes flagged transactions to Neo4j AuraDB (secrets from env vars) |

## Project layout
```
conf/base/            catalog.yml, parameters.yml (all hyperparameters and thresholds)
conf/mendeley/        Track B overrides: own sample file and output folder (`kedro run --env mendeley`)
conf/amlworld/        Track A overrides (`kedro run --env amlworld`)
conf/ablation_*/      Track B ablations: no card/device entities, no message passing
conf/longtrain*/      training-budget tests (more epochs, larger step)
data/01_raw ...       git-ignored data folders (kept via .gitkeep); outputs go to data/07_model_output/<env>/
src/fraud_ring_detector/
    pipeline_registry.py, settings.py, synthetic.py, amlworld.py
    pipelines/fraud_detection/{nodes.py, model.py, topology.py, pipeline.py}
docs/TRACK_A_PLAN.md   the plan and decision rules, written before seeing Track A results
notebooks/             older standalone walkthrough (simpler logic)
tests/                 26 tests: split, features, causality, adapters, patterns, windows, evaluation
tools/                 run_all.py, run_seeds.py, check_text_overlap.py, check_id_columns.py, check_pattern_match.py
filter_data.py         sampling + schema adapters (paysim / banking / amlworld)
profile_dataset.py     dataset triage before I build on anything
```

## Data
| Dataset | Track | Role | Sample file written by `filter_data.py` |
|---|---|---|---|
| Mendeley multi-pattern banking dataset | B | Primary for fraud scoring | `data/01_raw/banking_sample.csv` |
| IBM AMLworld HI-Small (Kaggle) + `HI-Small_Patterns.txt` | A | Primary for ring detection | `data/01_raw/amlworld_sample.csv` |
| Built-in mocks (`--synthetic`, `--synthetic-banking`, `--synthetic-aml`) | all | Smoke tests only; values are invented | per schema |
| PaySim (Kaggle) | legacy | Still supported; balance columns can leak the label | `data/01_raw/paysim_sample.csv` |

Each dataset has its own sample file and its own output folder, so results from different datasets can never overwrite each other.
Check any file before trusting it:
```bash
python profile_dataset.py "data/01_raw/<file>.csv" --nrows 0
python tools/check_id_columns.py "data/01_raw/<file>.csv" ip_address card_id device_id --entity customer_id
```
Columns that encode the label (`fraud_type`, `pattern_type`, `pattern_id`) are used for analysis only, never as features.

## Run it
```bash
python -m venv venv && source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q                                                # set PYTHONPATH=src first

# 1-minute smoke tests with mock data
python filter_data.py --synthetic-banking && kedro run --env mendeley
python filter_data.py --synthetic-aml     && kedro run --env amlworld

# Track B on the real file (8 GB laptop: keep whole histories for a subset of customers)
python filter_data.py --input "data/01_raw/Fraud_Detection_Dataset.csv" --customer-frac 0.3
#   -> the printed negative_sampling_fraction (0.3099 for this file) is already set in conf/mendeley/parameters.yml
kedro run --env mendeley
python tools/run_seeds.py --seeds 1 2 3 --env mendeley

# Track A on the real file (a contiguous time window keeps full graph density and the true fraud rate)
python filter_data.py --input "data/01_raw/HI-Small_Trans.csv" --window-start 0.5 --window-rows 300000
kedro run --env amlworld
```
Memory guide (measured on mock data, CPU): about 100k rows ≈ 1.1 GB peak, 300k rows ≈ 2.3 GB; 1M rows would need roughly 6 GB.
Outputs land in `data/07_model_output/<env>/`: `metrics.json`, `threshold_sweep.csv`, `syndicates.csv`, `account_roles.csv`,
`pattern_recall.csv`, `fraud_rings_dashboard.html`; seed summaries in `data/08_reporting/<env>/`.

### Everything in one go (saves compute and round trips)
```bash
python tools/run_all.py --aml-input "data/01_raw/HI-Small_Trans.csv" --windows 0.3 0.5 --row-windows 0.85
```
It builds each AMLworld window, runs the model, archives the outputs, optionally re-runs Track B, and writes
`data/08_reporting/final/final_summary.md` plus **one** file to share, `data/08_reporting/results_bundle.zip`. It's resumable
(finished windows are skipped; `--force` redoes them), and a window that fails is reported in the summary instead of stopping the rest.

## Syndicate analysis (what it does and doesn't do)
- **Direction and value:** merged directed graph (amounts summed per account pair), directed amount-weighted Louvain.
- **Overlap:** edge-based communities; accounts touching several are listed in `account_roles.csv`.
- **Roles:** originator / mule / collector / mixed from in/out flow. Heuristics, not ground truth.
- **Shapes:** directed cycles up to `max_cycle_len` hops, plus fan-in / fan-out / gather-scatter hubs.
- **Not covered:** temporal ordering of hops, multi-bank views, learned roles, device/IP links between flagged customers.

## Optional: Neo4j AuraDB
Set `neo4j.enabled: true` in `parameters.yml` and export `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
(newer Aura: username = database = instance ID). Never commit secrets (see `.env.example`).

## Originality
All code and text are my own; the paper is cited as inspiration only. `tools/check_text_overlap.py` scans the repo for verbatim
overlap with a source text.

## Tech stack
Kedro, PyTorch, PyTorch Geometric (SAGEConv), scikit-learn, NetworkX (Louvain), PyVis, Neo4j (optional).
