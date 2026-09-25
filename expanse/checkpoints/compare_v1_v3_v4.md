# Expanse model comparison

Same rows for every model (v1 training dev split) and the same generation items. **nats/char** is comparable across tokenizers; nats/token only within one tokenizer.

| metric | v1 | v3 | v4 |
|---|---|---|---|
| nats/char (rows all models cover) replay | 0.1656 | 0.0333 | 0.0323 |
| nats/char (rows all models cover) fly | 0.3931 | 0.3136 | 0.3178 |
| nats/char (rows all models cover) code | 1.1455 | 0.4733 | 0.4468 |
| nats/char (rows all models cover) bio | - | - | - |
| nats/char (rows all models cover) connectome | 0.1810 | 0.1241 | 0.1247 |
| nats/char (all rows) replay | 0.1656 | 0.0333 | 0.0323 |
| nats/char (all rows) fly | 0.3931 | 0.3136 | 0.3178 |
| nats/char (all rows) code | 1.1754 | 0.5147 | 0.4871 |
| nats/char (all rows) bio | 0.9958 | 1.3696 | 1.3583 |
| nats/char (all rows) connectome | 0.1820 | 0.1262 | 0.1267 |
| nats/token (all rows) replay | 0.4182 | 0.0841 | 0.0816 |
| nats/token (all rows) fly | 1.1983 | 0.9562 | 0.9690 |
| nats/token (all rows) code | 3.9057 | 1.8432 | 1.7444 |
| nats/token (all rows) bio | 4.9714 | 5.8276 | 5.7792 |
| nats/token (all rows) connectome | 0.6135 | 0.4254 | 0.4270 |
| exact-answer accuracy | 0.2000 | 0.3200 | 0.3200 |
| code pass rate | 0.0000 | 0.0000 | 0.0000 |
| bio token-F1 | 0.2850 | 0.2203 | 0.2303 |
| PubMedQA accuracy | 0.0000 | 0.0000 | 0.0000 |
| connectome exact match | 0.1600 | 0.3600 | 0.3600 |
| connectome token-F1 | 0.6738 | 0.8136 | 0.8173 |

Generation items per metric: {"exact": 25, "code": 25, "bio": 25, "pubmedqa": 25, "connectome": 25}; greedy, max 96 new tokens.
Scoring length / tokenizer per model: v1 128 (word), v3 128 (bpe), v4 128 (bpe).
