# Input-evoked CNS readout and reproducible experiments

Designed: 2026-09-25. Implemented: 2026-09-26. Status: regression tests passed; six-run exploratory pilot complete.

## Purpose

Improve Expanse's experimental architecture and the reliability of its conclusions on the existing CPU budget. The repository's CNS audit found that a constant response accounts for most of the graft's gain, and real wiring did not outperform matched rewiring. The next useful question is whether a graft that must transmit input-dependent information can learn a useful correction. A new architecture is a hypothesis, not evidence of better generated answers or unprecedented novelty.

## Selected change

Add `LabConfig(readout="evoked", state="reset")`. The existing `absolute` readout remains the default. Let `F(u)` be the current ungated CNS write, including its recurrent dynamics, output RMS normalization and projection. The new write is:

```text
h_out = h + gate(h) * (F(u(h)) - F(0))
```

The zero-drive response uses the same parameters, graph and recurrent steps. Both terms remain differentiable. A zero-drive column is evaluated alongside the real token columns, so the baseline does not depend on other tokens or examples. Normalize each response before subtraction; normalizing their small difference could amplify numerical noise. This is not batch-mean centering.

With disjoint input/output nodes and zero graph edges, the two responses coincide even with learned biases. Consequently the evoked write vanishes. This makes the zero-edge intervention a direct structural check. It does not establish that biological topology is superior: a shuffled graph can also transmit information.

Persistent-state evoked readout is rejected for now. Subtracting a state-dependent baseline would also remove stored-memory effects and needs its own design. Default config serialization omits the new default field to preserve existing run keys and historical receipts.

## Reliability changes

- Advance persistent decode position through padding-only chunks without changing neural state. Invalidate decode state when changing dtype/device.
- Treat the configured radius as initialization scaling, not a contractivity guarantee after unconstrained gains or edge weights learn.
- Bind new runs to the exact checkpoint, graph source, corpus and relevant source bytes. Refuse stale or unverified completed-run reuse instead of silently overwriting evidence.
- Record actual encoded training/evaluation row order and batch-plan hashes. Compare protocol, input provenance and available evaluation cohort hashes before issuing paired verdicts. Preserve historical results without fabricating missing provenance.

## Alternatives considered

Deeper recurrence and adaptive stopping are plausible separate studies, but the present experiment first resolves whether the existing branch conveys input information. Lowering the edge optimizer epsilon may enable learning, but combining that change with the readout change would confound attribution. A dense bottleneck adapter matched for non-edge parameters remains a required capacity control before a larger confirmatory study; the empty graph has different trainability and is only a structural control.

## Exploratory pilot

`X-CNS-002` compares absolute real, evoked real, and evoked shuffled readouts on the frozen v3 trunk, fixed edges, dynamic gates, a 3,000-type subgraph and identical data/order within each seed. Seeds 31 and 32, 40 core-only steps, batch 4, hash-selected 12 dev rows per source, and final zero-edge/gate-zero ablations. There are no temporal rows and no optimizer epsilon change. Its hypothesis file is locked before running. The pilot checks trainability, intervention behavior and runtime; two short runs per arm cannot establish a general language-model improvement or biological advantage. No model is promoted from this pilot.

Before a confirmatory study, choose at least five unused paired seeds, a sufficient training budget, a matched dense-adapter control, a separately frozen evaluation cohort and generation metrics. Define practical gain thresholds and per-source regression limits before observing those results. Report seed variation separately from row resampling uncertainty.

## Validation

Check unchanged legacy forward behavior and run identities; exact identity at zero gate; zero-drive and zero-edge cancellation; gradients through both terms against a dense reference; causality, batch independence and padding; checkpoint serialization; cached-state padding/dtype transitions; tampered protocols, cohorts and input bytes; and end-to-end runs on the real checkpoint. Recompute historical verdicts read-only to detect compatibility regressions.

## Research grounding

Sources checked on 2026-09-25. These papers motivate design constraints; none validates the proposed Expanse readout.

1. [Miller et al., Transformer Circuit Faithfulness Metrics are not Robust](https://arxiv.org/abs/2407.08734), CoLM 2024. Circuit conclusions depend on how ablation is performed. Therefore report the exact intervention and avoid equating zero-edge sensitivity with biological necessity.
2. [Scholkemper et al., Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs](https://arxiv.org/abs/2406.02997v3), ICLR 2025. Their analysis highlights how centering changes graph signals. The proposed zero-drive subtraction is a different, token-local intervention and must be tested empirically.
3. [Jolicoeur-Martineau, Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871), October 2025. Small recursive networks are a useful direction for structured reasoning tasks; puzzle results are not evidence of general conversational capability here.
4. [Are Your Reasoning Models Reasoning or Guessing?](https://arxiv.org/abs/2601.10679v2), March 2026 revision. Incorrect attractors in recursive reasoning motivate checking answer correctness rather than treating convergence as success.
5. [CHASE: Cache-Hole-Adapted Skip Exit for Looped State-Space Language Models](https://arxiv.org/abs/2607.10110v2), September 2026 revision. Skipping computation can leave stale recurrent state; this motivates explicit decode-state tests and deferring adaptive exits until measured and trained appropriately.

## Historical evidence re-audit

The stricter checker reproduces the recorded outcomes for H-CNS-001/002/003/005/006. H-CNS-004 is now
inconclusive on fresh evaluation because only one arm records final-row cohort hashes in seeds 2 and 3.
Its issued SUPPORTED verdict remains immutable; missing evidence is not repaired by inventing hashes.
Pairs missing both receipts retain legacy behavior, explicitly labeled `legacy_unverified` in fresh outputs.
The [machine-readable re-audit](checkpoints/cns_lab/evidence_reaudit_20260926.json) records the distinction.

## Pilot outcome

All six runs completed. The [generated report](checkpoints/cns_lab/X-CNS-002.report.md) and
[raw summary](checkpoints/cns_lab/X-CNS-002.report.json) bind the measured values to their metric-file hashes.
Every arm within each seed has identical input manifests, encoded row/order receipts and batch plans;
initial held-out loss is exactly equal. Each arm trains 277,920 parameters.

| Readout / topology | Mean development nats/char | Mean zero-edge loss increment | Mean token-varying write power |
|---|---:|---:|---:|
| Absolute / real | 0.425371 | 0.000842 | 10.24% |
| Evoked / real | 0.425105 | 0.006073 | 10.33% |
| Evoked / shuffled | 0.424951 | 0.006227 | 10.77% |

The evoked real arm is slightly better than the absolute real arm on both seeds, a roughly **0.06%**
mean loss reduction. Its zero-edge loss increment is about **7.2 times** as large; zero-edge and gate-zero
losses coincide within numerical precision, confirming that the evoked core cannot retain the old
zero-edge bias benefit. This structural property is partly enforced by the design, not an independently
discovered biological mechanism. The shuffled arm is better than real wiring on both seeds.

The change does not increase token-varying write power appreciably: almost 90% still sits in the mean
write. It therefore removes the zero-drive shortcut without demonstrating richer semantic computation.
The real evoked arm improves mean code, bio, connectome and fresh losses slightly, but worsens replay
from 0.074129 to 0.074495 nats/char. Two short seeds and 60 reused dev rows do not justify a performance
claim beyond this configuration. No generated-answer benchmark was run and no production checkpoint
was replaced or promoted.

Measured training times were approximately 0.88–1.00 seconds per step at four CPU threads. This is a
local runtime observation, not a portable speedup claim. The manifest binds project inputs and code;
installed numerical dependencies and hardware are not hermetically captured (runtime versions are
recorded separately).

Validation: the full suite passed **360 tests, 10 skipped**. After the final cache-completion and report
integer-validation changes, **32 focused tests passed**. Core coverage includes real v3 default parity,
evoked checkpoint/KV integration, dense-versus-sparse gradients through both terms, cancellation,
causality, padding-only chunks and dtype transitions. Compilation and `git diff --check` passed.
