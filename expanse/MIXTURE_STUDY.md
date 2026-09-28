# Source mixture and semantic robustness study

This upgrade makes continued training controllable and its results easier to judge. It adds source sampling, exact sampling resume, donor-preserving continuation, and a frozen generated-answer diagnostic. The experiment compares the current v4 parent with two short continuations. It never changes the checkpoint served in chat.

**Completed pilot:** both 80-update continuations finished. Balanced sampling
reduced pooled dev loss by 0.319% versus v4, with a fresh-source regression.
All three checkpoints scored 4/72 on strict final-answer accuracy and 0/12 fully
correct groups. Neither candidate was promoted. [Full results, raw replies and
validation](checkpoints/mixture-20260927-v2/REPORT.md).

## Why this experiment

The v3 training receipt inherited by v4 contains 41,986 packed training rows. Fresh examples and connectome facts supply about 71% of those rows; biomedical examples supply about 6%. A uniform shuffle of rows therefore does not provide equal exposure to the six sources. In addition, the default donor wake schedule is intended for newly grafted experts: applying it again during continuation resets already-active donor biases. `--wake preserve` now leaves those values untouched during both training and final evaluation.

[DoReMi](https://arxiv.org/abs/2305.10429) and [DoGE](https://arxiv.org/abs/2310.15393) motivate treating domain proportions as an explicit training variable. [DynaMiCS (May 2026)](https://arxiv.org/abs/2605.10770) studies mixture changes with constraints on other domains. This implementation provides fixed row probabilities and measures regressions; it does not implement those papers' learned mixture optimizers, and equal-source weights are a baseline rather than an established optimum.

[GSM-Symbolic](https://arxiv.org/abs/2410.05229) and [CheckList](https://aclanthology.org/2020.acl-main.442/) motivate controlled numeric changes and behavioral invariance tests. Our small diagnostic uses two paraphrases each of a base problem, a numeric contrast, and an operation contrast. It reports correctness together with consistency; an unchanged wrong answer is not evidence of understanding. These components combine established ideas for this project; no claim of unprecedented research is made.

## Implementation

- `train_expanse.py --sampling natural` retains the historical random-permutation ordering and incomplete-tail behavior. `--sampling balanced` selects a source by its row probability and cycles through independently shuffled rows within that source.
- `--source_weights replay=1,fly=1,code=1,bio=1,connectome=1,fresh=1` accepts explicit positive weights for every nonempty packed source. Omission gives equal probabilities in balanced mode. These are row probabilities, not token loss weights.
- `--sampling_seed` controls sampling without changing corpus split membership. The receipt records actual successful-update rows and supervised reply tokens by source, including repeated exposures.
- New partial checkpoints bind parent bytes, packed metadata, sampling settings, and training arguments. Resume restores sampler, optimizer, model and random states. Historical partials retain their old natural order but explicitly mark prior exposure as unverified.
- `evaluate_reasoning.py` freezes solver-labelled groups, screens the listed local corpora for normalized exact matches and recognized expression/loop matches, and retains every raw reply. Invalid, ambiguous, over-budget and failed generations remain in the denominator.
- A scalar or an explicit terminal answer marker is required. Intermediate reasoning is not verified. This strict format requirement can reject a numerically correct answer embedded in unsupported prose; raw replies make that inspectable.
- Strict final-answer accuracy jointly measures correctness, supported formatting and completion within the token budget. The prompts do not explicitly request that output format. Differing intermediate `x = ...` assignments are conservatively treated as conflicting markers, even when a trace could be correct; negation or alternative-answer wording can also cause rejection. Interpret invalid-answer reasons alongside the score.
- `run_mixture_study.py` binds code, parent, suite and corpus bytes, reconstructs each arm's feature cache from identical initial bytes, runs arms serially, and validates the resulting receipts. Completed artifacts cannot be silently reused after their hashes change.
- Chat now respects the loaded checkpoint's context length and asks for a shorter prompt instead of silently discarding its beginning. Decoding also skips the unused final forward pass.

## Frozen pilot

| Setting | Value |
|---|---|
| Parent | `checkpoints/supermix_expanse_v4.pt` |
| Arms | Natural row mixture; equal-source row mixture |
| Updates | 80 per arm, batch 4, sequence 128 |
| Trainable components | Trunk and graft parameters using the existing trainer; frozen encoders stay frozen |
| Learning rates | Trunk 0.00001; grafts 0.0001 |
| Seeds | Corpus and sampling 2026 |
| Corpus | v3 data, fly rows 1500, cap 12000 per source/fresh family |
| Dev subset | Stable 24-row cap per source; identical across arms |
| Donors | Preserve inherited state |
| Compute | Four CPU threads; serial arms |
| Generated diagnostic | 12 groups, 72 prompts, suite seed 20260927 |
| Decode | Native greedy, all grafts enabled, at most 64 new tokens within context |
| Selection policy | No automatic selection or promotion |

The suite scans 10 local JSONL files, including their train and held-out rows. It excludes six candidate groups for normalized exact overlap and one for recognized semantic overlap. Its recognizer covers literal integer expressions and a narrow Python add/subtract loop shape. Unknown phrasings, donor pretraining data and generated FlyCore rows remain outside semantic coverage; this is not proof of globally unseen problems.

Each arm processes 320 row exposures, less than one percent of the corpus. The study matches update counts and padded batch dimensions; supervised token counts, routed work and elapsed time can differ. One short seed cannot establish convergence, durable capability improvement or broad model quality. Report per-source loss alongside generated correctness; never substitute one for the other.

The actual continuation corpus has 41,984 rows, including 1,438 FlyCore rows.
FlyCore targets are regenerated from the loaded v4 model, so their text and hash
split can differ from the historical v3 receipt even with the same sampling seed.
Both arms use identical regenerated rows. FlyCore dev is a current-run diagnostic,
not a claim that those synthetic semantics were untouched throughout model history.
The legacy receipt phrase "frozen original FlyCore" refers here to the frozen
copy of the loaded parent, rather than an independently restored initial graft.

## Reproduce

Run from the repository root with an existing full v4 checkpoint and corpus:

```powershell
python expanse/evaluate_reasoning.py --make-suite expanse/checkpoints/reasoning-suite.json --seed 20260927 --groups-per-family 6
python expanse/run_mixture_study.py prepare --directory expanse/checkpoints/mixture-study --suite expanse/checkpoints/reasoning-suite.json --steps 80 --threads 4
python expanse/run_mixture_study.py train --directory expanse/checkpoints/mixture-study
python expanse/evaluate_reasoning.py --suite expanse/checkpoints/reasoning-suite.json --model parent=expanse/checkpoints/supermix_expanse_v4.pt --model natural=expanse/checkpoints/mixture-study/natural.pt --model balanced=expanse/checkpoints/mixture-study/balanced.pt --out expanse/checkpoints/mixture-study/reasoning.json --max-new-tokens 64 --threads 4
```

Freeze Python source files before preparation. To recover an interrupted arm, rerun the same `train` command. If source or input bytes change, preserve the old study and prepare a new empty directory. Model files and feature caches are ignored by Git; receipts, the suite, and reports can be reviewed separately.
