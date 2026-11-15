# Track A plan: can a graph model recover laundering patterns? (IBM AMLworld, HI-Small)

Written before the first real run, so that the decision rules cannot be bent to fit the results.

## 1. Problem framing
- **Business question.** Given account-to-account transfers, which transactions belong to a laundering attempt, and can we find
  whole attempts (rings), not just isolated transactions?
- **Unit of prediction.** One transaction. **Unit of success.** (a) transaction-level PR-AUC; (b) pattern-instance detection.
- **Why a graph?** Laundering is relational (fan-in, fan-out, cycles, layering). The honest question is whether a *learned* graph
  model adds anything over *hand-built* relational features (M2), which published AML work shows to be strong.
- **Non-goals.** Real-time scoring, regulatory claims, any statement about real banks.

## 2. Data understanding (what we know and what we must still check)
- Columns: Timestamp, From Bank, Account, To Bank, Account.1, Amount Received, Receiving Currency, Amount Paid,
  Payment Currency, Payment Format, Is Laundering; plus `HI-Small_Patterns.txt` (laundering attempts between BEGIN/END markers).
- First profile (first 400,000 rows only): 149 positives (0.037%), 134 of them in the last 20%. Lesson: **laundering is not spread
  evenly in time**, so any slice must be chosen consciously and its positives per split counted.
- Receiving accounts with two or more distinct senders: 30.8% of transactions but 77.2% of laundering transactions
  (about nine times the fraud rate). Relational signal exists; a degree feature may capture much of it.
- To check on the real file: time span; positives per period; share of laundering in each pattern type; whether the patterns file
  matches the transactions exactly (the loader reports matched instances).

## 3. Data preparation decisions
| Decision | Choice | Why | Cost |
|---|---|---|---|
| How to fit 8 GB | Contiguous **time window** of about 300k transactions | Keeps natural fraud rate and full graph density inside the window; no sampling bias in precision | Patterns straddling the window edge are cut; one period only |
| Alternative rejected | Random row sampling | Thins every account's history, breaks sequences and degrees | n/a |
| Account identity | Bank id + account id | Account ids may repeat across banks | none |
| `type` | Payment Format | Closest analogue of transaction type | none |
| Amounts | Amount Paid, plus amount received, currencies, cross-bank, same-currency, log ratio | No exchange rates are given | Amounts only roughly comparable |
| Pattern labels | Attached for evaluation only | Using them as features would leak the answer | none |
| Split | Chronological 60/20/20 inside the window | Mimics batch scoring | Test needs enough positives (the pipeline stops if a split has fewer than 5) |

## 4. Feature design (all causal: only past transactions)
- Tabular: log amount, payment format, currencies, cross-bank, same-currency, log amount ratio, hour of day.
- Account flow: what the sender received and sent *before* this row, what the receiver sent and received before, pass-through
  ratios (money in versus out, the signature of mules), reverse-transfer-seen (round-tripping).
- History: last-10 transactions of the sender (CNN input); in/out degree snapshots per split.
- No pattern labels, no bank-id categories with thousands of levels.

## 5. Models (a ladder, not a contest)
M1 tabular gradient boosting; M2 gradient boosting with hand-built graph/sequence features; M3 hybrid GraphSAGE + CNN.
Published AML baselines (gradient boosting with mined graph patterns; graph transformers) are the external reference points;
any comparison must use the same dataset variant, the same split logic and the same metric.

## 6. Evaluation design and decision rules (fixed in advance)
- Primary metric: PR-AUC on the held-out future period. Also precision, recall, alerts per 10,000 at a validation-chosen threshold.
- Uncertainty: sender-clustered bootstrap (200 resamples) for PR-AUC and for the paired differences M2-M1 and M3-M2.
- Pattern-level: for each pattern type, transaction recall and the number of instances with at least one / at least half of their
  test-period transactions flagged.
- **Decision rules.**
  1. If the 95% interval of (M3 - M2) includes 0, I report "no evidence the learned graph helps", not "it helps slightly".
  2. If M2 beats M3, the finding is "hand-built relational features win on this data", and I say so first.
  3. A pattern type with fewer than 5 test instances is reported as anecdotal.
  4. I repeat the run on at least two other windows (`--window-start 0.3` and `0.7`) before drawing any conclusion about stability.
  5. No claim about "detecting syndicates" unless instance-level detection (not only transaction recall) supports it.

## 7. Threats to validity
Synthetic data; a single window per run; pattern boundaries cut by the window; currency heterogeneity; the pipeline's
full-batch training limits scale; label timing (laundering labels are assigned by the simulator, not by an investigator);
multiple comparisons across pattern types.

## 8. Execution checklist
1. Download HI-Small_Trans.csv and HI-Small_Patterns.txt into `data/01_raw/`.
2. `python profile_dataset.py data/01_raw/HI-Small_Trans.csv --nrows 0` (or skip if RAM is tight).
3. `python filter_data.py --input data/01_raw/HI-Small_Trans.csv --window-start 0.5 --window-rows 300000`; read the printed
   window, positives and pattern counts before going on.
4. `kedro run --env amlworld`; then the same for two more windows.
5. Fill the results table; write the decision-rule outcome first, the story second.

## 9. Extensions after the first result
Neighbour-sampling mini-batches for larger windows; temporal ordering of hops in the syndicate analysis; learning roles;
a second AMLworld variant (LI-Small) as an out-of-distribution check.
