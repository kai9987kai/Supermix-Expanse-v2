# Expanse Connectome Laboratory — spec

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
