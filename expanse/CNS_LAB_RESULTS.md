# Expanse Connectome Laboratory — results

Pre-registered, matched-arm experiments on the male-CNS core of Supermix Expanse v3, run on one laptop CPU
(2026-09-24/25). Method and spec: [`CNS_LAB_DESIGN.md`](CNS_LAB_DESIGN.md). Code: `src/connectome_lab.py`, `cns_lab/`.

**Bottom line.** A trained CNS core helps Expanse (1.8% lower held-out loss than no core, every seed), but under
this protocol **the fly's real wiring does no better than a degree-preserving shuffle of it**, and **carrying the
CNS state across tokens does not help** on memory tasks. An independent audit reproduced every verdict exactly and
also showed *why* the wiring test came out null: the trained core behaves mostly like a learned constant bias, and
its per-edge weights could barely move — so these are nulls about this training regime, not proof that the fly's
wiring is useless. In the exploratory arms one change stood out — an **input-dependent gate** — and a pre-registered
follow-up confirmed it (H-CNS-004 SUPPORTED, 3/3 new seeds). It works mostly as an adapter: 83% of its gain needs
no graph, the rest needs *a* graph, and the fly's own wiring adds nothing over a degree-preserving shuffle
(H-CNS-005 barely SUPPORTED, H-CNS-006 NOT SUPPORTED). Built into the full model it gives **Expanse v4**: 2.5% lower
held-out loss than v3, 93% of it attributable to the gate.

## Pre-registered verdicts

Hypotheses were written, hash-locked (commit `83c98d0`) and committed before any lab run; the verdicts were issued
by `cns_lab/prereg.py` from the recorded metrics alone and are immutable (`cns_lab/verdicts/`).

| hypothesis | arms (3 paired seeds) | primary metric | locked rule | result | verdict |
|---|---|---|---|---|---|
| **H-CNS-001** real wiring beats a degree-preserving shuffle | real vs shuffled, learned Dale edges, full graph | pooled held-out nats/char | all seeds favour real, mean Δ ≥ 0.5% of control (0.00212) | mean Δ **+0.00006**; 1/3 seeds favour real | **NOT SUPPORTED** |
| **H-CNS-002** a trained core beats no core | real vs none, full graph | pooled held-out nats/char | all seeds, mean Δ ≥ 1% (0.00431) | mean Δ **+0.00795** (1.84%); 3/3 | **SUPPORTED** |
| **H-CNS-003** persistent CNS state beats per-token reset | persist vs reset, 3,000-type subgraph | held-out temporal-task nats/char | all seeds, mean Δ ≥ 2% (0.135) | mean Δ **−0.0068** (persist worse); 0/3 | **NOT SUPPORTED** |

Per seed (final eval, step 250; Δ > 0 favours the treatment):

| seed | H1 shuffled | H1 real | Δ | H2 none | H2 real | Δ | H3 reset | H3 persist | Δ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.42346 | 0.42321 | +0.00025 | 0.43118 | 0.42321 | +0.00797 | 6.7834 | 6.7968 | −0.0134 |
| 2 | 0.42329 | 0.42334 | −0.00005 | 0.43118 | 0.42334 | +0.00784 | 6.7640 | 6.7647 | −0.0007 |
| 3 | 0.42310 | 0.42313 | −0.00002 | 0.43118 | 0.42313 | +0.00805 | 6.7535 | 6.7598 | −0.0063 |

(The verdict files' `all_same_sign` field means "every seed favours the treatment"; for H-CNS-003 it is `false`
although all three deltas are negative.)

### What was matched

Within a seed every arm shared the base model (v3 with its CNS core removed), the core's seed-determined
initialisation (gate 0, so every arm starts as the same function), the 33,166 training rows and their order, the
optimiser and schedule, 250 steps, and the eval rows (60 v3-dev rows per source; + 160 held-out temporal rows for
H-CNS-003). The audit confirmed this directly: the step-1 training loss is bit-identical between paired arms, the
shuffle is redrawn per seed (7.2% edge overlap with the real graph, every type's degrees, input magnitudes and Dale
signs kept) and the runs are deterministic.

## Reading the results

**H-CNS-002 — a trained core helps.** +0.00795 nats/char (95% CI 0.0077–0.0082). Code carries 65% of the gain, bio
19%, connectome 8%, replay 5%, fresh 3%; the gain was still growing at step 250 (0.00696 at step 125). On the same
300 eval rows v3 with its own co-trained core scores 0.41174 against 0.43118 without a core, so the fresh 250-step
cores recover ~41% of what the original core was worth (shuffled-graph cores: 40.6%). The "4.5M trainable
parameters" are effectively the 684k non-edge ones (see 3 below).

**H-CNS-001 — no detectable wiring effect, and a tight one.** Paired mean +0.00006 (SD 0.00017, t = 0.62); 95% CI
−0.00036 to +0.00048. An equivalence test against the threshold rejects an effect of that size (p = 0.001): any
advantage of the real wiring is below ~0.0004 nats/char, about 4–6% of the core's own effect. The small positive
mean comes entirely from bio on seed 1. This has the same shape as QuixiAI/FlyGPT's null (mixed signs, mean far
below threshold), measured about 50× more tightly in absolute terms — but on a plug-in core inside a pretrained
model, not a connectome-only model.

**H-CNS-003 — no benefit from persistent state.** Persistent was worse on all three seeds, but not significantly
(t = −1.85, p = 0.21; 95% CI −0.023 to +0.009): the data exclude a benefit larger than ~0.004 nats/char, not a
harm. (The logged state autocorrelation, 0.955 persist vs 0.933 reset, compares separately trained cores and
mostly reflects how similar consecutive inputs are; it is not a memory measure.)

## Why the nulls are weak evidence about the fly (audit findings)

An independent four-lens audit (verdict recomputation, matching/leakage, sensitivity probe, statistics) with
adversarial verification found no error in the runs or verdicts, and these limits on what they mean:

1. **The trained core is mostly a constant bias.** With every edge weight set to zero no signal can reach the
   output types (the input and output type sets are disjoint), so the core's write becomes one constant vector.
   Evaluated on the full 300 eval rows, that constant alone keeps **79–83% of the trained core's gain**; the part
   that needs signal to pass through the graph is only **0.00170 / 0.00132 / 0.00141 nats/char** (seeds 1–3,
   mean 0.00148) — *below H-CNS-001's 0.00212 threshold*. So the protocol could not have supported H-CNS-001 even
   against a control graph that carried no information, and H-CNS-002 passes on the constant write alone (0.0063 >
   0.0043): it is not the positive control for wiring it was meant to be. The trained write is 98% its token mean
   (per-token cosine 0.9998) at 4.4% of the hidden-state size. The paired null still caps any real-wiring advantage
   at about a third of that small graph-dependent effect (shuffled cores do the same graph-dependent work, 0.00141).
2. **The cores do adapt to their own graph, but both graphs are equally learnable.** On an 80-row probe, swapping
   the shuffled graph into the trained real core costs 0.00153 nats/char (the reverse 0.00082), yet trained on its
   own graph each reaches the same loss. Linearised, both graphs give a full-rank input→output transfer (713) wider than the 320-d projection
   bottleneck, so the learned projections can absorb either.
3. **"Learned edges" never learned.** Multipliers stayed within 0.99–1.01 (std ~2e-4); 58–70% of the 3.83M edges
   moved by less than 1e-6 and about half not at all; resetting every multiplier to 1 changes loss by 2e-6, and the
   exploratory fixed-edge arm ended within 1.4e-6 of its learned-edge twin. The cause is the per-edge gradient size:
   ~1e-9 or less (every gradient inside the core is scaled by a gate that starts at 0 and ends near 0.025), below
   AdamW's eps of 1e-8, so updates stayed proportional to those tiny gradients instead of being normalised. The
   learning-rate budget would also have capped moves at ~±2.5%. H-CNS-001 therefore compared fixed real wiring with
   fixed shuffled wiring.
4. **H-CNS-003 measured answer format, not memory.** Neither arm learned the temporal tasks: exact match 0/160 in
   every run and task family, 9.5 nats/token (worse than a uniform guess over the vocabulary). 83% of the loss is
   choosing the wrong *kind* of token — the end-of-reply token alone carries 27%, and on recall the model puts 60%
   of its mass on space-prefixed digits while the answer digit is scored without a space. 87% of the persist-minus-
   reset difference sits on the end-of-reply token; the memory-relevant part is +0.0012 nats/char with mixed signs.
   The trunk's attention already sees the whole prompt, so the test never reached a regime where CNS memory could
   show.
5. **Smaller points.** The 3-seed sign rule cannot be significant on its own (best p = 0.125), so the magnitude
   threshold does the work; intervals are conditional on the fixed eval rows (per-row losses were not stored); the
   replay/bio eval rows are the first 60 dev rows in file order (replay = physics only) rather than a hash sample;
   a few held-out *pattern* prompts are prefixes of training prompts (no effect: exact match was 0 everywhere);
   `gate_by_source` carries no information for a static gate (it equals the mean gate for every source).
   Before locking, H-CNS-001's threshold was lowered from 1% to 0.5% and H-CNS-003's primary metric changed from
   exact match to loss (both before any lab run; only three 3-step smoke runs preceded them).

## Next experiments (most informative per CPU-hour)

0. ~~Confirm and attribute the dynamic-gate effect~~ — done: see *Follow-up* and *v4* below. What remains open
   there: whether the small graph-dependent part (H-CNS-005) needs a *graph* or just any extra input pathway of the
   same size (the empty arm's input side is untrainable), with more seeds given its marginal pass.
1. **Attribute the core effect.** Add a constant-bias / zero-weight core arm and a matched non-connectome core
   (same non-edge parameters, random or identity W), 3 seeds, and state any wiring effect relative to the
   graph-dependent part of the gain (zero-weight arm minus trained arm), not to the pooled loss.
2. **Test learned wiring for real.** Make the edges trainable in practice — a small AdamW eps (or rescaled θ) for
   the edge group and a much larger edge learning rate, with a pre-registered check that multipliers move past
   10% — top layers unfrozen (`train_scope=core+top`), ≥ 5 seeds, an equivalence margin, per-row losses saved.
3. **Force memory through the core.** Restrict the trunk's attention to a short window (or unfreeze the top layers)
   so information can only persist through the CNS state; score answer tokens only (not the end-of-reply token),
   per task, with an exact-match floor check; include `window:K` and `learned_decay`.

## Exploratory arms (X-CNS-001, one seed, no claims)

Same protocol as H-CNS-003 (3,000-type subgraph, learned Dale edges unless stated, reset state unless stated, seed 1,
250 steps); each arm changes one thing against the H-CNS-003 seed-1 reset run. Paired arms share the seed's
initialisation and data, so the relevant noise is the paired spread: about 0.00005–0.00017 nats/char pooled and
0.006 on the temporal metric (from the 3-seed pre-registered pairs) — read differences under ~3x that as noise.

| arm | what changes | pooled | Δ pooled | temporal | Δ temporal | temporal EM |
|---|---|---|---|---|---|---|
| reset (reference) | H-CNS-003 control, seed 1 | 0.42733 | — | 6.783 | — | 0% |
| **dynamic_gate** | input-dependent gate | **0.41965** | **−0.00768** | **2.599** | **−4.184** | **10%** |
| shuffled | degree-preserving shuffle | 0.42725 | −0.00008 | 6.799 | +0.015 | 0% |
| unsigned | all edges excitatory | 0.42731 | −0.00002 | 6.754 | −0.029 | 0% |
| residual_edges | W_bio (1 + α) edges | 0.42733 | +0.00000 | 6.784 | +0.000 | 0% |
| fixed_edges | edges not learned | 0.42733 | +0.00000 | 6.784 | +0.000 | 0% |
| window8 | state carried, reset every 8 tokens | 0.42734 | +0.00001 | 6.796 | +0.012 | 0% |
| tanh | tanh instead of relu | 0.42737 | +0.00004 | 6.782 | −0.001 | 0% |
| learned_decay | state carried with learned retention | 0.42738 | +0.00005 | 6.790 | +0.006 | 0% |
| random_sparse | uniform random graph, same size | 0.42744 | +0.00011 | 6.787 | +0.004 | 0% |
| random_strengths | measured strengths dealt at random | 0.42746 | +0.00013 | 6.810 | +0.026 | 0% |
| sqrt_indeg | count / sqrt(in-degree) weights | 0.42756 | +0.00023 | 6.958 | +0.175 | 0% |
| raw_counts | raw synapse-count weights | 0.42766 | +0.00033 | 7.003 | +0.220 | 0% |

* **The input-dependent gate is the one large effect** — as large again as the whole H-CNS-002 core effect on the
  pooled loss (bio −0.025, code −0.011 nats/char), the temporal loss falls from 6.78 to 2.60, and it is the only arm
  that answers any memory task (recall 12.5%, count 27.5%, pattern and first-item 0%). Its effective gate is ~0.53–0.62
  per token (highest on temporal prompts, 0.62; lowest on replay/fresh, 0.53) against 0.02 for the static gate.
  **This is very probably not the connectome:** the core's write is nearly constant (see audit point 1), so
  `h + (gate + tanh(W_g·h)) ⊙ write` is in effect a trainable 320→320 tanh adapter on the frozen trunk. It needs a
  zero-edge dynamic-gate control (and a shuffled one) before it says anything about the fly.
* **Normalisation matters for trainability:** the FlyGPT-style count/√in-degree weights and raw counts are clearly
  worse on the temporal metric (+0.18, +0.22) and slightly worse pooled; the input-fraction weights v3 uses are best.
* Everything else — shuffled or random wiring, unsigned edges, residual or fixed edges, tanh, window/learned-decay
  state — sits within about the noise band, consistent with the pre-registered nulls. Random strengths (+0.026
  temporal) and unsigned edges (−0.029) are at the edge of the band; one seed cannot separate them from noise.
* Edge multipliers stayed within 0.976–1.020 in every learned-edge arm (residual edges moved the most); the fixed-
  edge arm matched its learned-edge twin to 1e-6.

## Follow-up: the dynamic gate (H-CNS-004..006, pre-registered)

Locked (commit `0e3e8ce`) after the exploratory result and before any of their runs; same 3,000-type subgraph
protocol; **seeds 2–4**, so the seed-1 observation does not count. New control: `topology="empty"` — the same cell
types with no edges, so with a dynamic gate the core is a pure input-dependent adapter (its output-type biases start
at +0.1: at 0 the relu output types get no gradient and a pre-lock smoke run showed the control dead; the gate still
starts at 0 and every other initial value matches the real arm).

| hypothesis | control → treatment (pooled nats/char, seeds 2 / 3 / 4) | mean Δ | threshold | verdict |
|---|---|---|---|---|
| **H-CNS-004** dynamic beats static gate | 0.42715→0.41961 / 0.42704→0.42047 / 0.42710→0.41997 | **+0.00708** (1.66%) | 0.00427 | **SUPPORTED** 3/3 |
| **H-CNS-005** real wiring beats empty graph (dynamic gate) | 0.42004→0.41961 / 0.42236→0.42047 / 0.42125→0.41997 | **+0.00120** | 0.00105 | **SUPPORTED** 3/3 (barely) |
| **H-CNS-006** real wiring beats shuffle (dynamic gate) | 0.41966→0.41961 / 0.42065→0.42047 / 0.41979→0.41997 | **+0.00001** | 0.00105 | **NOT SUPPORTED** 2/3 |

| arm (mean of seeds 2–4) | pooled | temporal nats/char | temporal exact match |
|---|---|---|---|
| static gate, real wiring | 0.42709 | 6.75 | 0% |
| dynamic gate, empty graph | 0.42122 | 2.75 | 7.7% |
| dynamic gate, shuffled | 0.42003 | 2.65 | 8.3% |
| dynamic gate, real wiring | 0.42002 | 2.62 | 9.4% |

**Reading (independent audit, verdicts reproduced exactly).**
* **H-CNS-004 is solid:** 1.66% lower pooled loss (paired 95% CI 0.0059–0.0083), mostly bio and code; temporal
  loss 6.75 → 2.62, largely by fixing the answer format, and the only arms that answer memory prompts.
* **Where the gain comes from:** 83% of it (0.0059) is reached with **no graph at all** (empty arm); the remaining
  17% (0.0012) needs a graph, and a **degree-preserving shuffle reproduces all of it** — the fly-specific part is
  about zero (+0.00001; any real-wiring advantage below ~0.0005, equivalence to the threshold shown).
* **H-CNS-005 passes by a hair** (0.00015 above threshold; not significant seed-by-seed, t(2) = 2.84, one-sided
  p = 0.05; seed 2 alone is below threshold; the gap shrank from step 125 to 250). The threshold assumed static-gate
  noise; the dynamic arms are 4–7× noisier. The +0.1 bias start cannot explain the gap (identical step-1 loss and
  gradients; the per-token normalisation divides it out), but the empty arm's whole input side is untrainable, so
  what H-CNS-005 licenses is *"an input-dependent write routed through a (any, degree-matched) graph adds ~0.001"*,
  not *"the fly wiring helps"* — its locked claim text says more than the data do.
* **Mechanism:** the static core adds a near-constant vector (4.6% of the hidden-state size, 0.07% of its power
  token-dependent). The dynamic gate scales it per token and channel by `tanh` of a low-rank projection of the
  frozen trunk's state (6–20 directions hold 52–83% of its energy): a 320→320 tanh adapter writing at ~15% of the
  hidden size with ~45% token-dependent power.
* Minor: `gate_abs_mean` records only the static gate vector, so it understates the dynamic arms' effective gate
  (~10× larger; see `gate_by_source`); H-CNS-004's static seeds 2–3 are reused H-CNS-003 runs recorded before the
  per-row / write diagnostics existed (verdict unaffected).

## Expanse v4: v3 + the dynamic CNS gate

`build_v4.py` / `run_v4.sh`. v3's trained full-graph core (11,751 types, 3.83M edges) is converted into a lab core
— checked **bitwise identical on real rows** (max |Δlogit| 0.0) — and the dynamic gate is added at zero, so v4
starts as exactly v3. Only the core trains (trunk frozen, edges fixed), 600 steps on the v3 corpus (fly rows not
included), new gate parameters at lr 1e-3 and v3's core parameters at 1e-4. A **matched control** trains the same
core the same way with the static gate.

| held-out nats/char (500 v3 dev rows, never trained on) | v3 | control (static gate, same training) | **v4** |
|---|---|---|---|
| pooled | 0.41016 | 0.40938 (−0.00077) | **0.39970 (−0.01045, −2.5%)** |
| code | | −0.0024 | **−0.0294** |
| bio | | −0.0010 | **−0.0141** |
| fresh / replay / connectome | | ≈ 0 | −0.0023 / −0.0018 / −0.0002 |

So **~93% of v4's gain is the gate**, not the extra training. On the standard comparison (v1's dev rows,
`checkpoints/compare_v1_v3_v4.md`):

| | v1 | v3 | **v4** |
|---|---|---|---|
| replay nats/char | 0.166 | 0.033 | **0.032** |
| code nats/char (all rows) | 1.175 | 0.515 | **0.487** |
| bio nats/char (all rows) | 0.996 | 1.370 | 1.358 |
| connectome nats/char | 0.182 | 0.126 | 0.127 |
| fly nats/char | 0.393 | **0.314** | 0.318 |
| fresh-number problems, exact answer | 20% | **32%** | **32%** |
| connectome exact match / token-F1 | 16% / 0.674 | 36% / 0.814 | 36% / **0.817** |
| biomedical token-F1 | **0.285** | 0.220 | 0.230 |
| code pass rate, PubMedQA | 0% | 0% | 0% |

v4 is better than v3 on code, replay and bio loss and bio F1, level on generation accuracy, and slightly worse on
fly rows (not in v4's core training). v1 still has the best bio numbers (its `<unk>`-heavy word vocabulary makes
rare terms cheap; see the v3 notes). Checkpoint: `checkpoints/supermix_expanse_v4.pt` (loads with
`expanse_core.load_expanse`; the lab core's config and anatomy are stored in it).

## Files

* `cns_lab/hypotheses/*.json` + `.lock.json` — the locked hypotheses; `cns_lab/verdicts/*.verdict.json` — verdicts
* `checkpoints/cns_lab/runs/<run key>/metrics.json` — every run's metrics (trained cores `core.pt` are not in git)
* `checkpoints/cns_lab/analysis/summary.md` — all runs; `<run key>.md` — edge drift, pathways, hubs, modularity,
  leak by superclass, gate by source per run
* `checkpoints/cns_lab/bench.md` — backend benchmark
