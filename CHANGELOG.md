# Changelog

## 0.5.4: README in a friendlier voice
- README rewritten to read like a person explaining a project: TL;DR, the story, plain-language takeaways, a short section on how
  this differs from Cai (2026). Every number, table and command is unchanged; the technical reference now sits under its own heading.
  No code changes.

## 0.5.3: the training-budget test overturns an earlier conclusion
- `longtrain_no_graph` (300 epochs, step 0.01, no neighbour messages) on the Track B data: M3 reaches PR-AUC 0.729 (0.671 to 0.785),
  level with M2's 0.725; M3 minus M2 = +0.003 (0.001 to 0.007); account takeover recall 0.00 -> 0.95. The earlier finding that
  the network trails the trees was a training-budget artifact (60 full-batch epochs = 60 weight updates).
- Whether learned message passing adds anything once training is adequate is untested (`longtrain`, slow); the Track A M3 results are
  at the same small budget and need the same caveat (`amlworld_longtrain` environment added).
- `tools/run_all.py`: `--aml-env`, `--track-b-env` and `--final-dir` run a different environment into its own folder and bundle,
  so a new experiment cannot overwrite earlier results.

## 0.5.2: .gitignore fix
- `data/**/*` also ignored the sub-folders themselves, so Git could not re-include the `.gitkeep` placeholders and the `data/`
  folder structure was never actually tracked. Added `!data/**/` so the placeholders are kept while data, results and secrets stay ignored.
  (It never affected results: the code creates folders as it needs them.)

## 0.5.1: final Track A results and a test of M3's training budget
- Three valid AMLworld windows are in (time 0.3, time 0.5, rows 0.85): M2 beats M1 and M3 at equal alert volume in all three, and the
  syndicate layer, even on M2's flags, finds no group of three or more accounts and no cycles. README records the outcome against the
  pre-registered decision rules.
- Training-budget finding: M3 is trained full-batch (60 epochs = 60 weight updates), ran all 60 epochs in Track B and window 0.3, and in
  rows 0.85 stopped at epoch 11 with its best validation score at epoch 1 (test ROC-AUC 0.44). Its weakness may partly be undertraining.
- `model.warmup_epochs`: epochs that can never be selected as best and never count toward patience.
- Environments `longtrain_no_graph` (300 epochs, step 0.01, no messages) and `longtrain` (150 epochs, with messages) on the Track B data.
- 26 tests pass.

## 0.5.0: guards against degenerate windows, and syndicates from the best model
- Found with the real run: AMLworld's activity thins out after about hour 297, so a clock-based window starting at 0.7 held 374
  transactions (59% laundering) and every model scored 1.000. `filter_data.py` now refuses windows holding under half the requested
  rows (`--allow-small-window` overrides), `--window-by rows` places a window by transaction count so it always lands in dense data,
  and `tools/run_all.py` accepts `--row-windows` and excludes degenerate windows from its summary.
- The syndicate analysis can use the flags of M2 instead of M3 (`syndicates.topology.score_source`, `graph_features` in the amlworld
  environment). With M3's flags the real windows produced only two-account "syndicates", no groups and no cycles.
- README records the two valid Track A windows and the equal-volume Track B recalls. 25 tests pass.

## 0.4.8: one-command evaluation
- `tools/run_all.py`: builds every AMLworld window, runs `kedro run --env amlworld`, archives outputs, optionally re-runs Track B,
  and writes one combined summary table and one zip to share. Resumable; failures are recorded, not fatal.

## 0.4.7: tie-safe equal-volume comparison
- Found with the real window's output: M1 reported the same threshold (about 105 alerts per 10,000) for every budget from 5 to
  50, and M2 repeated one value, because many transactions share exactly the same score and a threshold cannot cut inside a
  tie group. Validation-calibrated thresholds now cut only between distinct scores (and allow no alerts if even the top tie
  group exceeds the budget).
- New `at_equal_volume` metrics: flag the top-ranked test transactions up to each alert budget (ties broken at random), so every
  model is compared at the same volume. `pattern_recall.csv` now uses these flags (`alert_budget_per_10k` column).
- Validation-calibrated results stay in `at_alert_budget`, with the realised alert rate beside the target (it drifts on the
  test period). 22 tests pass.

## 0.4.6: equal-operating-point comparison and the first Track A result
- First real AMLworld window (27 hours, 306,128 transactions, 504 laundering rows): M1 0.045, M2 0.151, M3 0.036 PR-AUC;
  M3 minus M2 = -0.114 (-0.177 to -0.064). 40% of laundering rows are unlisted in the patterns file (diagnosed with
  `tools/check_pattern_match.py`: not a matching bug). README updated.
- `evaluation.alert_budgets_per_10k`: recall and precision at equal alert volume, threshold chosen on validation and applied to
  the test period; the realised alert rate is reported next to the target. `evaluation.pattern_alert_budget` (20 in the
  amlworld environment) scores pattern recall at that budget, so models are compared at the same operating point.
- 22 tests pass.

## 0.4.5: pattern-match diagnostic
- `tools/check_pattern_match.py`: classifies UNCLASSIFIED laundering rows as matching problems (same pair in the patterns file),
  partial overlaps, or genuinely unlisted laundering. First real window: 40% of laundering rows were unclassified.

## 0.4.4: configuration that survives updates
- `evaluation.negative_sampling_fraction` now lives in the dataset's environment (`conf/mendeley/`, the two ablations: 0.3099;
  `conf/amlworld/`: 1.0). Base keeps the neutral default of 1.0, so unzipping a new version no longer overwrites an edited value.
- `filter_data.py` reports how many laundering transactions matched a line in the AMLworld patterns file, and warns below 50%.

## 0.4.3: ablation results
- README records the Track B ablations. Without card/device information M2 falls from 0.725 to 0.201 PR-AUC; without message
  passing M3 is unchanged within noise (0.646 vs 0.656); the hybrid network trails gradient boosting in every variant.
  No code changes.

## 0.4.2: Track B corrected results and two ablations
- First run with the sampling correction and bootstrap intervals (negative_sampling_fraction 0.3099, population fraud rate
  0.149%): M1 0.128, M2 0.725, M3 0.646 PR-AUC; M3 minus M2 = -0.081 (95% interval -0.118 to -0.052). README updated.
- `model.message_passing` (default true): false swaps GraphSAGE for a node MLP with identical inputs and sizes.
- Environments `ablation_no_entities` and `ablation_no_graph`, each with its own output folder, to separate the value of
  shared card/device information from the value of learned message passing.
- 21 tests pass.

## 0.4.1: Windows encoding fix
- `fraud_rings_dashboard` is now saved with `encoding: utf-8` (base, mendeley and amlworld catalogs). Windows opens text files
  as cp1252 by default, which cannot encode some characters inside the inlined PyVis libraries, so the last node failed with
  a `'charmap' codec can't encode characters` error. Linux and macOS never showed it. Nothing else changes.

## 0.4.0: Track A (ring detection on IBM AMLworld) and sharper uncertainty
- `src/fraud_ring_detector/amlworld.py`: schema detection (`Account.1`), laundering-pattern parser (BEGIN/END blocks),
  post-hoc pattern labels, time-window selection, and the mapping to the pipeline schema (bank + account ids, payment format as
  type, currency/cross-bank/amount-ratio extras). Pattern labels are excluded from features.
- `filter_data.py`: detects the schema, writes one sample file per dataset (`paysim_` / `banking_` / `amlworld_sample.csv`),
  `--window-start/--window-rows` (AMLworld time window), `--patterns`, `--synthetic-aml`.
- Per-dataset environments: `conf/mendeley/` and `conf/amlworld/` redirect the sample file and write outputs to
  `data/07_model_output/<env>/` so results from different datasets never overwrite each other. `tools/run_seeds.py` follows.
- New causal `account_flow_features` (sender receipts/sends before, receiver sends/receipts before, pass-through ratios,
  reverse-transfer-seen). Off by default; on in the amlworld environment. Track B results are unchanged.
- `pattern_evaluation_node`: recall per laundering pattern type; instances detected-any / detected-majority.
- Sender-clustered bootstrap: 95% intervals for PR-AUC and for the paired deltas M2-M1 and M3-M2 (`evaluation.bootstrap_samples`).
- Metrics label for M3 no longer mentions card/device nodes (AMLworld has none).
- Mock generator fixes found while testing: pattern start times were bunched and capped too early, which left no laundering
  in the test period. The pipeline's fraud-rows-per-split guard caught it, and its message now says how to fix real data.
- `docs/TRACK_A_PLAN.md`: plan and decision rules written before any real result.
- README rewritten in first person with the Track B results. 20 tests pass.

## 0.3.1: README neutrality, RAM-friendly sampling, id checker
- README rewritten Track-B-first and dataset-neutral (PaySim demoted to a legacy/smoke-test option); two-track table added.
- `filter_data.py --customer-frac F`: keeps every transaction of all customers with fraud plus a fraction F of the others,
  so sampled customers keep complete histories (better than row sampling for graph and sequence features); prints the exact
  `evaluation.negative_sampling_fraction` to use.
- `tools/check_id_columns.py`: uniqueness and cross-customer sharing of id-like columns (e.g. `ip_address`), reading only
  the named columns.
- Measured memory on mock banking data: 100k rows ~1.1 GB, 300k rows ~2.3 GB peak (1M rows extrapolates to ~6 GB).
- 14 tests pass.

## 0.3.0: Track B (broader fraud scoring) and a stricter experiment design
Motivated by the first real-data A/B run on the Mendeley-style file, which showed that the earlier pipeline did not test
the graph claim: no balance columns existed (so dropping them changed almost nothing), the baseline used far fewer columns
than a plain profiler model (0.44 vs 0.965 PR-AUC), card/device sharing (58% of fraud) was not in the graph, and precision
was reported at a 4.4% sampled prevalence instead of the real ~0.15%.
- **Bug found while designing this:** message passing was directed sender -> receiver, so on customer -> merchant data the
  sender embedding received no neighbour messages (SAGEConv aggregates source -> target). New `features.bidirectional_edges`
  (default true) adds reverse edges. This also changes PaySim results relative to 0.2.x.
- **Baseline ladder:** M1 = gradient boosting on ALL tabular columns (auto-detected numeric extras and low-cardinality
  categoricals; label-like and id-like columns excluded); M2 = M1 + causal shared-entity counts + hand-built graph and
  sequence features; M3 = hybrid GraphSAGE+CNN. The hybrid must beat M2, not just M1.
- **Entity nodes:** `features.entity_columns` (e.g. card_id, device_id) become graph nodes with edges to the customers who used
  them, built from past transactions only, plus causal "distinct customers on this card/device so far" features.
- **Honest reporting:** `evaluation.negative_sampling_fraction` converts precision and alert volume to the real population
  (`population_precision`, `alerts_per_10k`); recall per `fraud_type` (post-hoc, never a feature); `tools/run_seeds.py`
  runs several seeds and writes mean +/- std.
- `filter_data.py` keeps all columns, adds `--synthetic-banking`, and defaults to 0.03 negative sampling only for files
  (use `--legit-frac 1.0` when the file fits in memory: row sampling thins every account's history).
- `conf/mendeley/parameters.yml` + soft-merged parameters: `kedro run --env mendeley`.
- `profile_dataset.py`: `--nrows 0` reads the whole file, excludes id-like numeric columns, prints the feature list, and
  warns when positives concentrate in one time split (a first-rows slice of a time-ordered file is unrepresentative).
- Metrics keys changed: `baseline_tabular_full`, `baseline_graph_features`, `hybrid_sage_cnn`. 13 tests pass.

## 0.2.3: banking-schema adapter and dataset triage
- `filter_data.py` now maps banking-style schemas (`customer_id`, `merchant_id`, `transaction_type`, `timestamp`) to the
  pipeline schema, derives `step` (hours since first transaction), tolerates snake_case headers, and fills neutral balance
  columns when a dataset has none (then set `features.drop_balance_features: true`).
- `fraud_type` is carried through for post-hoc analysis only: it encodes the label and must never be a feature.
- Transaction-type one-hot features are now derived from the data instead of PaySim's hard-coded list.
- New `profile_dataset.py`: positives per time split, entity repeat rates, shared devices/IPs, tabular-baseline strength and
  single-column leakage suspects. Run it before building on any dataset.
- Tests for the adapter (9 tests pass).

## 0.2.2: leakage switch
- New parameter `features.drop_balance_features`. PaySim cancels fraudulent transfers, so balance columns can leak the
  label; set it to `true` to drop every balance-derived feature (transaction, sequence and node level).
  Run and report BOTH settings. Added a test for it.

## 0.2.1: syndicate analysis upgrade and originality hardening
- **Louvain limits addressed** (new `topology.py`): directed amount-weighted graph and directed Louvain; overlapping
  memberships via edge-based communities; flow-based roles (originator / mule / collector); cycle and fan-hub mining.
- New outputs: `account_roles.csv`; richer `syndicates.csv` (roles, overlap, cycles). Node renamed
  `louvain_clustering_node` -> `syndicate_analysis_node`. New `syndicates:` block in `parameters.yml`.
- Measured trade-off documented: lowering the topology candidate threshold finds more cycles but wrecks precision.
- Renamed `HybridGCFD` -> `HybridGraphCNN` and removed paper-specific phrasing so nothing implies a replication.
- Added `tools/check_text_overlap.py`; scanned all repo files against the paper text: zero shared 5/6/8-word sequences.
- Added `tests/test_topology.py` (6 tests total pass). The standalone notebook keeps the simpler undirected Louvain.

## 0.2.0: modelling and engineering overhaul

### Why
v0.1 ran, but nothing was learned or measured: features, temporal tensor and evaluation labels were random, there was
no training loop, and the Neo4j and Kedro wiring failed. v0.2 makes the pipeline actually train and evaluate.

### Model and evaluation fixes
| Problem in 0.1 | Change in 0.2 |
|---|---|
| `torch.randn` node features | Real node features (degrees, amounts, drain rate) from past transactions only |
| `torch.randn` temporal tensor | Real last-K sequence per originating account (`build_features`) |
| No training (one forward pass) | Full-batch training, focal loss, AdamW, early stopping on validation PR-AUC |
| Random ground truth in evaluation (`torch.rand_like`) | Real `isFraud` labels |
| Account-level scores | Transaction-level (edge) scores |
| Bipartite `SAGEConv` with one `x` (index-out-of-range risk) | Single homogeneous account graph with a shared node index |
| `nrows=50000` head slice (early, almost no fraud) | Stratified sample keeping every fraud row (`filter_data.py`, `prepare_dataset`) |
| No time awareness | Chronological train/val/test split on `step`; node snapshots built per split |
| ROC-AUC only | PR-AUC primary, plus P/R/F1 at validation-chosen threshold, threshold sweep |
| No baseline | Gradient-boosting baseline |
| Hardcoded dropout/hyperparameters in code | All in `conf/base/parameters.yml` |
| "EAF1" and "topological recall" claims | Removed; `f1` is plain F1, cycle count dropped from metrics (see README) |

### Kedro and configuration fixes
| File | Change |
|---|---|
| `src/pipelines.py`, `src/nodes.py` | Moved to `src/fraud_ring_detector/pipelines/fraud_detection/{pipeline,nodes,model}.py` |
| (missing) | Added `pipeline_registry.py`, `settings.py`, `__init__.py` files, `pyproject.toml` |
| `conf/base/catalog.yml` | Removed the bogus `fraud_graph_db` CSV entry; added outputs (JSON/CSV/HTML) so nodes are pure and Kedro does the I/O |
| `params:neo4j_credentials` | Removed: `params:` never reads `credentials.yml`. Secrets now come from env vars |
| `run_louvain_community_detection` wrote files and HTML directly | Returns the HTML string; the catalog writes it |
| `.gitignore` | Keeps `data/**/.gitkeep` and `conf/local/.gitkeep`, ignores `.env`, `.viz/`, egg-info |
| `requirements.txt` | Added `kedro-datasets[...]`, `numpy`; removed unused `duckdb`; `neo4j>=5.20` |

### Neo4j fixes
- Username and database are the instance ID on newer Aura (not `neo4j`); `database_` is no longer hardcoded.
- `CREATE` the relationship (the old `MERGE` collapsed repeated transactions and overwrote `amount`).
- Batched writes, uniqueness constraint, `try/finally` for `driver.close()`, optional and off by default.
- No credentials in code, notebooks or config.

### Other
- `filter_data.py`: chunked stratified sampling, `--synthetic` smoke-test mode, writes `paysim_sample.csv`.
- Added `tests/test_pipeline_smoke.py` and `notebooks/fraud_detection_model.ipynb` (standalone walkthrough, no secrets).
- README rewritten: removed unsupported claims, added limitations, run instructions and a results table to fill in.
