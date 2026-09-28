# Matched source-mixture pilot, 27 September 2026

Training and generated-answer evaluation are complete. **Neither continuation
improved strict final-answer accuracy: all three checkpoints scored 4/72.**
Equal-source sampling slightly improved dev loss but regressed on fresh examples.
No candidate was selected or promoted; the chat continues to serve v4.

The primary metric is **strict final-answer accuracy**. It jointly requires a
correct number, a supported terminal answer format, and completion within the
token budget. Prompts do not explicitly request that output format. Conflicting
intermediate `x = ...` assignments, negation and alternative-answer wording may
cause rejection even if a reader can identify the right answer. The frozen score
must be read with invalid-answer counts and raw replies; it is not a pure measure
of mathematical reasoning. Any manual reply inspection is descriptive only.

## Generated answers

| Metric | Parent v4 | Natural | Balanced |
|---|---:|---:|---:|
| Strict correct answers | 4/72 (5.56%) | 4/72 (5.56%) | 4/72 (5.56%) |
| Arithmetic | 0/36 | 0/36 | 0/36 |
| Python traces | 4/36 | 4/36 | 4/36 |
| Fully correct six-prompt groups | 0/12 | 0/12 | 0/12 |
| Correct base/contrast pairs | 0/48 | 0/48 | 0/48 |
| Consistent paraphrase pairs, including wrong answers | 13/36 | 14/36 | 12/36 |
| Parsed but numerically wrong answers | 63 | 62 | 62 |
| Invalid answers, including budget exhaustion | 5 | 6 | 6 |
| Generation errors | 0 | 0 | 0 |

Both candidates got exactly the same four items correct as the parent: no newly
correct or lost-correct items. Natural changed 14 raw replies and balanced changed
12, but none of those changes improved this score. More consistent wrong replies
do not demonstrate better reasoning.

Invalid-answer breakdown:

| Reason | Parent v4 | Natural | Balanced |
|---|---:|---:|---:|
| No supported unambiguous final number | 2 | 4 | 2 |
| Generation budget exhausted | 3 | 2 | 3 |
| Ambiguous or negated wording | 0 | 0 | 1 |

Most failures were parsed, numerically wrong answers rather than format
rejections. Descriptive examples from the parent, without changing the scorer:

- `What is 736 / 8?` has answer 92. The model replied
  `60 / 8 = 10, 16 / 8 = 2, total 12`, changing the input and giving a wrong result.
- `x = 36; for i in range(3): x = x + 9` has final value 63. The reply instead
  carried out three subtractions and ended with `x is 9, total 9`.
- `What is 67 + 3?` has answer 70. The reply ended in `total 610`.

These cases identify input-number and operator fidelity as concrete weaknesses
to investigate. This short mixture pilot did not resolve them. The 72 prompts
are a narrow diagnostic, not an estimate of all mathematical or coding ability.
No benchmark answers were used in either continuation run.

Every raw reply and token ID is in [reasoning.json](reasoning.json), whose
canonical SHA-256 is `a0e534c55ccfeaf5a3559bfbb2ea3abd9d241c65c0f98805bf3accf3fcc6c49a`.
The [recomputation audit](evidence-audit.json) checks checkpoint,
source, suite and corpus hashes, recomputes scores and verifies complete groups.

## Training result

Equal-source sampling reduced pooled dev reply-token cross-entropy by 0.319%
against the parent and 0.390% against the natural-mixture continuation. Fresh
examples regressed. This is a small loss change on 24 dev rows per source, not
evidence of improved generated answers or general reasoning.

| Source | Parent | Natural | Balanced | Balanced minus parent |
|---|---:|---:|---:|---:|
| Pooled | 1.238227 | 1.239114 | 1.234276 | -0.003950 |
| Replay | 0.109643 | 0.108714 | 0.103095 | -0.006548 |
| Fly | 0.950854 | 0.942550 | 0.935843 | -0.015011 |
| Code | 1.896643 | 1.905309 | 1.896495 | -0.000149 |
| Bio | 5.610878 | 5.615759 | 5.610216 | -0.000662 |
| Connectome | 0.367936 | 0.370535 | 0.367857 | -0.000079 |
| Fresh | 0.034704 | 0.035334 | 0.035560 | +0.000856 |

Units: nats per supervised reply token. Lower is better. The initial evaluation
dicts are identical between arms. Parent, packed metadata, tokenizer, dev keys,
corpus settings, optimizer settings, seeds and donor-state policy match.

## Measured exposure

| Source | Natural rows | Balanced rows | Natural reply tokens | Balanced reply tokens |
|---|---:|---:|---:|---:|
| Bio | 16 | 58 | 283 | 1,099 |
| Code | 37 | 68 | 1,085 | 2,169 |
| Connectome | 85 | 48 | 1,778 | 967 |
| Fly | 8 | 39 | 252 | 1,234 |
| Fresh | 148 | 59 | 5,665 | 2,434 |
| Replay | 26 | 48 | 833 | 1,888 |
| Total | 320 | 320 | 9,896 | 9,791 |

Balanced sampling processed 1.061% fewer supervised tokens. Equal row
probabilities do not imply equal token exposure, and finite samples need not
contain equal numbers of rows. Fresh supplied 57.25% of natural-arm reply tokens
and 24.86% of balanced-arm tokens; bio supplied 2.86% and 11.22%, respectively.

Both arms trained 43,675,553 parameters in the trunk and grafts, retaining
88,366,363 frozen parameters. All 32 donor experts stayed active and their
recorded mean routing biases were unchanged. The corpus contained 41,984 train
and 144 dev rows, with no overlength drops. Fly targets derive from the current
v4 parent, so their dev split is not a historical noncontamination guarantee.

Wall time was 22.80 minutes for natural and 15.98 minutes for balanced, including
preparation and evaluation. The machine was under varying memory pressure and
the natural arm overlapped a regression test run. These timings are not a speed
benchmark. Each arm saw only 320 row exposures and one seed; neither result is
a convergence claim.

## Evidence and validation

- [Immutable study manifest](manifest.json): `5452afb23a106856d1f0407c2975a5660d96b2655b95adfdf9bc5b498a78e558`.
- Parent checkpoint SHA-256: `82d0163cb8b733635827df222067ae6de275437c2ff5ec8b6b6bc29479536067`.
- Packed data SHA-256: `126cebccfbcfb6cf37e0ce106834c38a8a3279ca4771ac23bd57f5fc46f5525c`.
- [Frozen 72-prompt suite](../reasoning-20260927-v2.json): `6df109c9788af112a658f002c9962d167369491ed488418829c1f8f5d18d8ec1`, scorer `strict-final-number-v2`.
- [Natural receipt](natural.receipt.json), [balanced receipt](balanced.receipt.json), and [completed-arm hashes](natural.run.json) ([balanced](balanced.run.json)).
- Full regression suite: **482 passed, 10 skipped**, with one existing tensor-to-scalar warning. Exact sampler/optimizer/dropout continuation is covered for both modes.
- Live chat API returned the expected impulse answer. After reloading the context fix, a 603-token prompt received the explicit 128-token-context message without model inference.

The first study preparation was stopped before any optimizer update to fix the
scorer's multiplication-marker ambiguity. Its original files remain in
`../mixture-20260927/`, including an abort note. The replacement suite was frozen
before either completed arm trained and before any generated-answer inspection.

See [the design, research and commands](../../MIXTURE_STUDY.md) for methodology
and the bounded corpus-overlap screening limitations.
