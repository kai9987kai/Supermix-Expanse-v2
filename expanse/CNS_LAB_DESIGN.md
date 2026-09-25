# Expanse Connectome Laboratory — spec

> **Status (2026-09-24):** implemented (`src/connectome_lab.py`, `cns_lab/`, tests `tests/test_connectome_lab.py` 29 passed,
> `tests/test_cns_lab.py` 7 passed). Hypotheses H-CNS-001..003 and the exploratory X-CNS-001 were **locked** (commit
> 83c98d0) before any lab run; the sweep (`bash expanse/cns_lab/run_sweep.sh`) runs 15 pre-registered runs, then 13
> exploratory ones. Results: `checkpoints/cns_lab/analysis/summary.md`, verdicts: `cns_lab/verdicts/`.
> See "As built" at the end for what changed from the plan below.

Purpose: turn Expanse's male-CNS core from "a biological graft" into a **controlled, pre-registered experiment** that can show whether the fly's wiring matters, following the method of QuixiAI/FlyGPT (fixed topology, one learnable value per real edge, degree-preserving scrambled control, pre-registered verdict) at Expanse's scale.

Repo: `C:/Users/kai99/Desktop/supermix-expanse/Supermix-expanse`, local branch `expanse-cns-lab` (from `expanse-v3-subword`; do NOT push, do NOT switch branches). Base model for lab runs: `expanse/checkpoints/supermix_expanse_v3.pt` (v1 schema + BPE tokenizer; `ec.load_expanse` loads it). Existing core: `expanse/src/connectome_full.py` (`build_full_graph`, `CSRMatmul`, `FullConnectomeCore`, `rewired_graph`) — read it first. CPU only, fp32, 16 GB RAM; `torch.set_num_threads(2)` in tests; no long jobs (the orchestrator runs benchmarks and lab runs).

## A. `src/connectome_lab.py` — `ConnectomeLabCore` (agent A)

`LabConfig` (dataclass, `to_dict`/`from_dict`, validated) — each field is one experimental axis:

| field | values | meaning |
|---|---|---|
| `topology` | `real` · `shuffled` · `random` · `none` | real male-CNS type graph · degree- and sign-stratified rewire (`connectome_full.rewired_graph`) · uniform random sparse graph with the same node count, edge count and per-edge magnitude multiset · no core (the graft is absent) |
| `sign_mode` | `dale` · `unsigned` | presynaptic neurotransmitter sign · all edges +1 |
| `strength_init` | `bio` · `random` | measured magnitudes · the same magnitudes randomly permuted across edges (seeded) |
| `edge_param` | `fixed` · `residual` · `dale_softplus` | no per-edge parameters (v3 behaviour) · `W_e = W_bio,e·(1+α_e)`, α=0 · `W_e = s_e·|W_bio,e|·softplus(θ_e)/softplus(θ₀)`, θ=θ₀=ln(e−1) so the multiplier starts at exactly 1 and the sign can never flip |
| `norm` | `frac` · `sqrt_indeg` · `raw` | input fraction (v3) · synapse count / sqrt(in-degree) (FlyGPT) · raw synapse count. Every mode is then globally rescaled so the |W| spectral radius equals `radius` (0.9), so modes differ in *shape*, not scale |
| `activation` | `relu` · `tanh` | v3 · FlyGPT |
| `state` | `reset` · `persist` · `window:K` · `learned_decay` | fresh state every token (v3) · state carried across tokens · carried, reset every K positions · carried with a learned per-type retention ρ=sigmoid(·) applied between tokens (init ρ≈0.5) |
| `microsteps` | int (v3: 4) | recurrent steps per token |
| `gate` | `static` · `dynamic` | per-channel gate (v3) · `static + tanh(W_g·RMSNorm(h))`, W_g zero-init (input-dependent routing, exact at birth) |
| `subgraph` | `None` or N | restrict to N cell types: strongly-connected core by synapse mass, keeping proportional input/output sets (for tractable lab runs); record the kept type names |

Update rule per token t (per position, all columns batched as in v3), for microstep k:
`drive = g_in ⊙ (W @ (g_out ⊙ r)) + U_t + b`; `r ← (1−a) ⊙ r + a ⊙ act(drive)`; readout from output types, RMS-normalised, `out_proj`, gated write. With `state=reset` r starts at 0 each token (v3); otherwise r carries from the previous position (sequential scan over T; PAD positions carry the state unchanged). Causality must hold in every mode.

Backends (`backend` arg): `csr_slab` (existing CPU path) · `torch_sparse` (`torch.sparse.mm`, CUDA if available) · `dense` (tests only). Per-edge parameters need gradients w.r.t. edge values: implement a custom autograd `EdgeSpMM(values, x)` → `W(values) @ x` whose backward gives `grad_x = W(values)ᵀ @ grad_y` and `grad_values[e] = Σ_c grad_y[post_e,c]·x[pre_e,c]` computed in edge chunks (bounded RAM, e.g. ≤256 MB per chunk). gradcheck in double on tiny graphs.

Requirements:
1. `lab_from_full(full_core, cfg) -> ConnectomeLabCore`: copies a trained `FullConnectomeCore`'s parameters; with the v3-equivalent config (`real, dale, bio, fixed, frac, relu, reset, 4, static`) the output must be **bitwise/1e-6 identical** to the full core (test on the real v3 checkpoint's core). `dale_softplus`/`residual` with their zero/θ₀ init must also be exact at birth.
2. `fresh_lab_core(hidden_size, graph, cfg, seed)`: new core (in/out projections, gains, leak, bias freshly and deterministically initialised by `seed`; gate zero → the graft is exact identity at birth).
3. Decode: when the hook sees a cached single-token step (`past_length>0`), a persistent-state core continues from the state it stored at the previous step (batch 1 is enough); at `past_length==0` it starts from zeros. Test: greedy KV-cached decode logits == full-forward logits (≤1e-4) for `persist` and `learned_decay`.
4. **Anatomical mapping in the checkpoint**: the core keeps int/float buffers `edge_src`, `edge_dst` (type index), `synapse_count` (int32), `bio_weight` (signed, float32) and, in its `spec()` dict (saved with the checkpoint), `node_id` (type names), `superclass`, `nt`, `n_neurons`, `input_nodes`, `output_nodes`, `subgraph`. `export_anatomy(core) -> dict` adds `learned_weight` and `learned_multiplier = learned/bio` per edge; `python -m connectome_lab export <ckpt> <out.npz|.parquet-free>` writes it.
5. Integration: `expanse_core.ExpanseModel` gains `attach_lab_core(model, core)` (replaces or removes `cns_full`; `topology=none` removes it) and the hook passes `token_mask` and `past_length`; `save_expanse`/`load_expanse` round-trip a model with a lab core (spec rebuilt from the saved `LabConfig` + anatomy). Existing v1/v3 checkpoints still load unchanged.
6. `bench_connectome.py`: backends × {full graph, subgraph 3000, 1500} × {edge_param fixed/dale_softplus} × {state reset/persist} → s/forward, s/backward, peak RAM at batch 8×128; writes `checkpoints/cns_lab/bench.json` + `.md`.
7. Tests `tests/test_connectome_lab.py`: exactness vs v3 core, exact birth for each edge_param, Dale signs never flip under dale_softplus after optimizer steps, EdgeSpMM gradcheck, causality for every state mode, decode==full forward for persistent modes, shuffled/random keep degree/edge-count invariants, sign/strength ablations do what they say, subgraph keeps inputs and outputs non-empty, anatomy export shapes and multiplier==1 at birth.

## B. `expanse/cns_lab/` — experiment harness (agent B)

1. **Pre-registration** `cns_lab/prereg.py` + `cns_lab/hypotheses/*.json`. A hypothesis file: `id`, `claim`, `arms` (name → LabConfig overrides), `control_arm`, `treatment_arm`, `seeds`, `steps`, `primary_metric` (e.g. `val_nats_per_char_pooled`), `direction` (`lower_is_better`), `success_rule` (`all_same_sign: true`, `min_mean_delta` absolute, or `min_mean_delta_rel` relative to the control mean), `secondary_metrics`, `notes`. `prereg.py lock <file>` writes `<id>.lock.json` = sha256 of the file + UTC time + git commit; `run_lab` refuses to run an unlocked or modified hypothesis. `prereg.py verdict <id>` reads only the recorded run metrics and prints/writes an immutable verdict: **SUPPORTED** (rule met), **NOT SUPPORTED** (all runs complete, rule not met), **INCONCLUSIVE** (runs missing/failed/NaN); refuses if the hypothesis hash changed since locking. Verdict file includes the per-seed table.
2. **Matched runs** `cns_lab/run_lab.py --hypothesis ID --arm NAME --seed S`: load the base checkpoint (v3); replace its CNS core with `fresh_lab_core(...)` for the arm (or remove it for `none`); **freeze everything except the lab core** by default (`--train_scope core|core+top|all`); identical data (v3 corpus via `train_expanse.load_corpus` with `v3=True`, seed-fixed order), optimizer (AdamW; lab core lr 1e-3, per-edge params lr 3e-4 like FlyGPT), schedule, step count and eval points across arms; eval at fixed steps (e.g. 25%, 50%, 100%) on a fixed validation set: per-source nats/char and pooled nats/char, plus throughput and peak RAM, gate mean |g|, dynamic-gate mean by source, state autocorrelation (persistent modes), leak distribution by superclass. Writes `checkpoints/cns_lab/runs/<ID>/<arm>/seed<S>/metrics.json` (+ the trained core only, not the whole model, to save disk). Resumable/skip-if-done.
3. **Orchestrator** `cns_lab/run_all.py --hypotheses H-CNS-001,...` runs every (hypothesis, arm, seed) once (arms shared across hypotheses are reused when their config and steps match), sequentially, then calls the verdicts; logs to `external/logs/cns_lab.log`.
4. **Analysis** `cns_lab/analyze.py`: for learned-edge runs — learned-multiplier histogram (text histogram + json; png if matplotlib is present), multiplier by pre/post superclass and by pathway (sensory→central→descending), top strengthened/weakened edges and hubs (with cell-type names), sign flips (must be 0 for dale_softplus), role modularity before/after, weighted in/out strength change; leak by superclass; dynamic gate by source; persistent-state autocorrelation. Writes `checkpoints/cns_lab/analysis/<run>.md` + `.json`.
5. **Temporal tasks** `cns_lab/temporal_tasks.py`: small synthetic held-out sets that need memory within the 128-token window (key→value recall at a distance, repeated-pattern continuation, count-back), scored by exact match, for the persistent-state hypothesis; deterministic by seed.
6. Tests `tests/test_cns_lab.py`: lock/verdict logic (hash tamper → refuse; all three verdict outcomes on synthetic metrics), run_lab smoke on a tiny config (few steps, subgraph) if feasible within the time limits, analysis on a synthetic core.

## Pre-registered hypotheses (the orchestrator writes and locks these BEFORE any lab run)

* **H-CNS-001** — real topology + learned Dale-preserving edges (`dale_softplus`) beats the degree-preserving shuffle with learned edges. SUPPORTED iff all seeds favour real and mean Δ ≥ 1% of the control's pooled nats/char.
* **H-CNS-002** — a trained CNS core (real, learned edges) beats no core (`topology=none`). Same rule.
* **H-CNS-003** — persistent CNS state beats per-token reset (real, learned edges): primary metric temporal-task exact match (higher is better; all seeds same sign and mean Δ ≥ 2 points), val nats/char secondary.
Exploratory (reported, no claims): unsigned edges, random strengths, random sparse topology, `sqrt_indeg` vs `frac` norm, `tanh` vs `relu`, `fixed` vs `dale_softplus`.

## As built

**Scale.** Measured with `cns_lab/bench_connectome.py` (`checkpoints/cns_lab/bench.md`; core alone, 8 x 128 tokens, 8 threads):
on the full graph a learned-edge step costs 5.9 s (1.7 s with fixed edges) and a persistent-state step 9.3 s; the custom
CSR path is 10-30x faster than `torch.sparse` on this CPU and the only one that fits the full graph. A whole lab step
(frozen v3 trunk + core) measured 4.7 s on the full graph and 7.8 s for the persistent scan on a 3,000-type subgraph.
Five to ten seeds x four full-scale arms is out of reach on this laptop, so the pre-registration uses **3 seeds x 250
steps**, core-only training:

| protocol | used by | graph | data |
|---|---|---|---|
| `P_full` | H-CNS-001, H-CNS-002 | full male-CNS type graph (11,751 types, 3,830,931 edges) | v3 corpus (replay, code, bio, connectome, fresh; v3's split seed 2026, so eval rows are v3 dev rows it never trained on) |
| `P_sub` | H-CNS-003, X-CNS-001 | 3,000-type subgraph (629,875 edges; synapse-mass core + proportional inputs/outputs) | v3 corpus, 2 of every 8 rows synthetic temporal-memory rows |

Both: AdamW, core lr 1e-3, per-edge lr 3e-4 (FlyGPT), 5% warm-up, cosine to 5%, clip 1.0, batch 8, eval at steps 125
and 250 on 60 dev rows per source (+160 held-out temporal rows for `P_sub`), `edge_param = dale_softplus` unless the
arm says otherwise.

**Hypotheses as locked** (rules are the verdict's only input):

* **H-CNS-001** real wiring vs degree-preserving shuffle (both learned Dale edges), `P_full`; SUPPORTED iff all 3 seeds
  favour real and mean delta >= 0.5% of the shuffle's pooled nats/char.
* **H-CNS-002** trained real core vs no core, `P_full`; all seeds and >= 1% (positive control: if this fails, H-CNS-001
  cannot be informative).
* **H-CNS-003** persistent vs reset CNS state, `P_sub`; primary metric held-out temporal-task nats/char, all seeds and
  >= 2%; temporal exact match and pooled loss secondary.
* **X-CNS-001** (no claims, seed 1): fixed edges, shuffled, random sparse, unsigned, random strengths, sqrt(in-degree)
  and raw-count normalisation, tanh, residual edges, dynamic gate, window:8, learned decay — against the `P_sub` reset arm.

**Deviations from the plan above.**
* Degree normalisation "learned" is the per-type input gain `g_in` every variant already trains; "Sigma|W|" is `frac`.
* Every graph variant is scaled to the same |W| spectral radius (0.9) so arms differ in structure, not gain; `frac` on
  the real/shuffled graph keeps its exact v3 values (rows sum to 0.9).
* The persistent-state hypothesis uses temporal-task *loss* as its primary metric (exact match is too coarse for 250
  core-only steps on a frozen trunk) and is run on a subgraph (cost).
* Runs restart rather than resume mid-run (each is 10-40 min).
* Expert routing is not tracked: the trunk (and so its MoE router) is frozen in every arm.
* Caveat for H-CNS-002: the v3 trunk was trained with its own core, so "no core" is a damaged v3, and part of any gain
  is the fresh core re-learning that role.

### Tooling changes after the first sweep

Asked for by the audit of the first pre-registered sweep (`CNS_LAB_RESULTS.md`, "Why the nulls are weak evidence").
All of it is backward compatible: for every locked hypothesis `prereg.run_key` returns the same keys (checked against
the original `prereg.py` for all 31 (arm, seed) pairs), an existing protocol trains and evaluates bit for bit as before
(checked against the original `run_lab.evaluate` and `ConnectomeLabCore`, forward and gradients), new behaviour needs a
**new** protocol key (which also gives the run a new key), and metrics are only added, never redefined.

* **Verdicts** (`prereg.py`, `verdict_schema_version: 2`; `schema` keeps its v1 string). The locked rule key
  `all_same_sign` stays and is documented as what it always meant: *every seed favours the treatment*; the verdict also
  reports it as `all_favour_treatment`. Before a run's metric counts, the run must be provably the run asked for:
  `run_key` equals its directory, arm / seed / full `LabConfig` match, not an `--unlocked` run, the recorded lock sha is
  the hypothesis' lock (a run shared with another hypothesis: that hypothesis' lock and the same arm config), and a
  trainable run's last eval is at the protocol's final step; anything else is a problem (-> INCONCLUSIVE). `inputs`
  records the sha256 of every metrics.json read. Issued verdict files are never touched; re-evaluating H-CNS-001..003
  with the hardened rule from the recorded runs reproduces the issued verdicts exactly.
* **New optional protocol keys** (`run_lab.py`):
  `eps_edge` (AdamW eps of the edge group only; the other groups keep 1e-8 -- per-edge gradients ~1e-9 sit below the
  default eps, which is why the edges did not learn), `eval_sampling: "hash"` (+ `eval_salt`: each source's dev pool
  sorted by a salted hash of `row_key(user, assistant)` instead of file order), `final_ablations: true` (at the final
  eval, pooled nats/char with every edge weight zeroed -- a constant write -- and with the gate closed:
  `val_nats_per_char_pooled_zero_edges`, `val_nats_per_char_pooled_gate0`, `graph_dependent_gain` = zeroed - trained,
  `core_gain` = gate closed - trained; the core is restored bit for bit) and `temporal.split_version: 2`
  (`temporal_tasks.split_rows`: also drops train rows whose prompt starts with a held-out prompt and `pattern` rows
  sharing a held-out repeating unit, up to rotation; the held-out rows are unchanged).
* **Always recorded now** (additive): `final_rows` -- per final-eval row `[nats, chars, tokens]` per source in eval-row
  order (+ `final_rows_sha256` of the row keys to check pairing, and `final_rows_temporal_answer`), so paired tests can
  be row-level; per eval `temporal_answer_nats_per_char` (reply tokens without the end-of-reply token, per reply
  character), `temporal_nats_per_char_by_task`, `temporal_answer_nats_per_char_by_task`, and with a core
  `write_rms_ratio` (RMS of the gated write / RMS of the hidden state at the graft) and `write_token_var_frac` (share
  of the write's power that varies across tokens; ~0 = a constant bias); `optimizer_eps` per param group.
  `temporal_nats_per_char` and every other existing key are unchanged.
* **Core** (`connectome_lab.py`): `_stats` adds the two write diagnostics on no-grad forwards only; forward numerics
  unchanged. `gate_by_source` is kept but is only informative for `gate="dynamic"` (a static gate gives every source
  the same mean |g|).
* **Analysis** (`analyze.py`): `summary.md` lists the ablation / write diagnostics of runs that recorded them, and an
  "Exploratory arms" section per exploratory hypothesis: every arm's final pooled / per-source / temporal numbers,
  the delta against the reference arm (`reset` if present, else the first arm), flagged when |delta| > `--noise_k`
  (2) x a noise floor -- `--noise_floor` if given, else per metric the SD over seeds of the paired treatment - control
  delta of the pre-registered hypotheses with the same protocol (for `P_sub`: H-CNS-003). Single seed, no claims.
* For the next pre-registration these make the audit's suggestions expressible: `eps_edge` (with a larger `lr_edge`
  and a pre-registered check that multipliers move), `final_ablations` (state a wiring effect relative to
  `graph_dependent_gain`), `eval_sampling: "hash"`, and `temporal.split_version: 2` with
  `temporal_answer_nats_per_char` as the memory metric.
* Tests: `tests/test_connectome_lab.py` 33 (31 passed + the 2 v3-checkpoint tests, which skip without the
  checkpoint), `tests/test_cns_lab.py` 24 passed; the run_lab tests use a tiny synthetic LM around a real lab core on
  a synthetic graph (no checkpoint, no corpus).
