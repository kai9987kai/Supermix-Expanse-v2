# X-CNS-002 exploratory pilot

Short development pilot; no confirmatory verdict, generation-quality claim, or biological-topology superiority claim. Seed SD is descriptive, not a confidence interval.

All compared arms have matching recorded input, data-order and evaluation-cohort receipts.

| Arm | Seed | Final nats/char | Graph-dependent gain | Core gain | Seconds/step |
|---|---:|---:|---:|---:|---:|
| absolute_real | 31 | 0.424988 | 0.000815 | 0.006190 | 0.902004 |
| evoked_real | 31 | 0.424806 | 0.006373 | 0.006373 | 0.905293 |
| evoked_shuffled | 31 | 0.424675 | 0.006503 | 0.006503 | 0.926422 |
| absolute_real | 32 | 0.425755 | 0.000870 | 0.005423 | 0.941802 |
| evoked_real | 32 | 0.425405 | 0.005773 | 0.005773 | 0.884046 |
| evoked_shuffled | 32 | 0.425226 | 0.005952 | 0.005952 | 0.998453 |

Graph-dependent gain = zero-edge loss minus trained loss; core gain = gate-zero loss minus trained loss. Positive values favor the trained core. Neither is a real-versus-shuffled comparison.

| Arm | Mean nats/char | Across-seed SD |
|---|---:|---:|
| absolute_real | 0.425371 | 0.000543 |
| evoked_real | 0.425105 | 0.000424 |
| evoked_shuffled | 0.424951 | 0.000390 |

No model was promoted. Full per-source losses, pairing and metric-file hashes are in the adjacent JSON.
