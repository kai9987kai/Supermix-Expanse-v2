# Lab run 12f2acbabde7eb00 — H-CNS-001 / real / seed 1

11,751 types, 3,830,931 edges. Config: `{"topology": "real", "sign_mode": "dale", "strength_init": "bio", "edge_param": "dale_softplus", "norm": "frac", "activation": "relu", "state": "reset", "microsteps": 4, "gate": "static", "subgraph": null, "radius": 0.9, "graph_seed": null, "backend": "csr_slab"}`

## Edge drift R = W_trained / W_bio

| min | p01 | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 0.9921 | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0002 | 1.0087 |

mean R 1.0000; edges changed > 10%: 0.00%; sign flips: 0

log2 R histogram:

```
 -0.0115 ..  -0.0100 |                                          15
 -0.0100 ..  -0.0085 |                                          51
 -0.0085 ..  -0.0070 |                                          173
 -0.0070 ..  -0.0055 |                                          444
 -0.0055 ..  -0.0040 |                                          1,192
 -0.0040 ..  -0.0025 |                                          3,094
 -0.0025 ..  -0.0010 |                                          11,275
 -0.0010 ..  +0.0005 | ######################################## 3,784,225
 +0.0005 ..  +0.0020 |                                          22,691
 +0.0020 ..  +0.0035 |                                          4,727
 +0.0035 ..  +0.0050 |                                          1,846
 +0.0050 ..  +0.0065 |                                          746
 +0.0065 ..  +0.0080 |                                          309
 +0.0080 ..  +0.0095 |                                          110
 +0.0095 ..  +0.0110 |                                          30
 +0.0110 ..  +0.0125 |                                          3
```

### By pathway (pre role -> post role)

| pathway | edges | geometric-mean R | std log2 R |
|---|---|---|---|
| central->central | 2,646,900 | 1.0000 | 0.0000 |
| central->input | 216,811 | 1.0000 | 0.0000 |
| central->output | 204,315 | 1.0000 | 0.0006 |
| input->central | 355,580 | 1.0000 | 0.0004 |
| input->input | 85,086 | 1.0000 | 0.0002 |
| input->output | 46,940 | 1.0001 | 0.0016 |
| output->central | 193,523 | 1.0000 | 0.0000 |
| output->input | 32,977 | 1.0000 | 0.0000 |
| output->output | 48,799 | 1.0000 | 0.0006 |

### Most changed superclass pairs

| pre -> post | edges | geometric-mean R |
|---|---|---|
| vnc_sensory->vnc_motor | 1,088 | 1.0004 |
| ascending_neuron->vnc_motor | 4,260 | 1.0002 |
| vnc_sensory->descending_neuron | 1,133 | 0.9998 |
| cb_intrinsic->cb_endocrine | 941 | 1.0001 |
| cb_sensory->descending_neuron | 2,151 | 0.9999 |
| visual_projection->descending_neuron | 5,096 | 1.0001 |
| vnc_intrinsic->vnc_efferent | 2,565 | 1.0000 |
| cb_sensory->cb_sensory | 2,065 | 1.0000 |
| vnc_sensory->vnc_sensory | 2,091 | 1.0000 |
| ascending_neuron->cb_motor | 1,470 | 1.0000 |
| ascending_neuron->descending_neuron | 28,989 | 1.0000 |
| sensory_ascending->vnc_intrinsic | 3,295 | 1.0000 |
| vnc_motor->vnc_motor | 996 | 1.0000 |
| sensory_ascending->ascending_neuron | 1,248 | 1.0000 |
| vnc_sensory->ascending_neuron | 7,722 | 1.0000 |

### Most strengthened edges (synapse count >= median)

| pre | post | synapses | W_bio | W_trained | R |
|---|---|---|---|---|---|
| aPhM2a (cb_sensory) | GNG014 (cb_intrinsic) | 1577 | +0.08483 | +0.08557 | 1.009 |
| LLPC1 (visual_projection) | DNp03 (descending_neuron) | 110 | +0.00531 | +0.00535 | 1.008 |
| SNxx31 (vnc_sensory) | ENXXX226 (vnc_efferent) | 163 | +0.01021 | +0.01029 | 1.008 |
| AN04B023 (ascending_neuron) | DNp28 (descending_neuron) | 3 | +0.00771 | +0.00777 | 1.007 |
| IN23B016 (vnc_intrinsic) | MNad09 (vnc_motor) | 593 | +0.03122 | +0.03145 | 1.007 |
| SNpp28 (vnc_sensory) | hg2 MN (vnc_motor) | 52 | +0.00430 | +0.00433 | 1.007 |
| ANXXX084 (ascending_neuron) | ENXXX226 (vnc_efferent) | 139 | +0.00871 | +0.00877 | 1.007 |
| AN18B002 (ascending_neuron) | MNml83 (vnc_motor) | 4 | +0.00538 | +0.00542 | 1.007 |
| DNge137 (descending_neuron) | ENXXX226 (vnc_efferent) | 1880 | +0.11775 | +0.11861 | 1.007 |
| INXXX288 (vnc_intrinsic) | MNad49 (vnc_motor) | 201 | +0.06199 | +0.06244 | 1.007 |
| AN09B002 (ascending_neuron) | DNg37 (descending_neuron) | 115 | +0.00675 | +0.00680 | 1.007 |
| AN19B004 (ascending_neuron) | Ta depressor MN (vnc_motor) | 3 | +0.00136 | +0.00137 | 1.007 |

### Most weakened edges (synapse count >= median)

| pre | post | synapses | W_bio | W_trained | R |
|---|---|---|---|---|---|
| aPhM2a (cb_sensory) | GNG099 (cb_intrinsic) | 650 | +0.08016 | +0.07952 | 0.992 |
| BM (cb_sensory) | DNge038 (descending_neuron) | 163 | +0.02659 | +0.02639 | 0.992 |
| SNxx20 (vnc_sensory) | EN00B004 (vnc_efferent) | 186 | +0.03174 | +0.03150 | 0.992 |
| aPhM2a (cb_sensory) | GNG140 (cb_intrinsic) | 486 | +0.06943 | +0.06890 | 0.992 |
| INXXX261 (vnc_intrinsic) | MNad54 (vnc_motor) | 846 | -0.29557 | -0.29331 | 0.992 |
| ANXXX084 (ascending_neuron) | EN00B004 (vnc_efferent) | 669 | +0.11416 | +0.11329 | 0.992 |
| LC10a (visual_projection) | AOTU019 (cb_intrinsic) | 14210 | +0.20985 | +0.20829 | 0.993 |
| SAxx01 (sensory_ascending) | INXXX386 (vnc_intrinsic) | 211 | +0.03272 | +0.03248 | 0.993 |
| aPhM4 (cb_sensory) | GNG099 (cb_intrinsic) | 721 | +0.08891 | +0.08826 | 0.993 |
| IN13A022 (vnc_intrinsic) | TTMn (vnc_motor) | 987 | -0.14807 | -0.14698 | 0.993 |
| SApp10 (sensory_ascending) | DNg06 (descending_neuron) | 228 | +0.03077 | +0.03055 | 0.993 |
| aPhM2a (cb_sensory) | GNG206 (cb_intrinsic) | 552 | +0.18614 | +0.18481 | 0.993 |

### Hubs (change of weighted strength, sum |W|)

* hubs_in_strength: gained AOTU025 (+0.0028), EN00B023 (+0.0025), AOTU006 (+0.0024), MNad69 (+0.0023), EN00B027 (+0.0022); lost DNge153 (-0.0027), SNpp23 (-0.0025), PVLP098 (-0.0024), AOTU019 (-0.0024), LgLG1b (-0.0022)
* hubs_out_strength: gained BM_InOm (+0.0074), LC10a (+0.0065), LC10d (+0.0063), IN10B011 (+0.0060), DNpe034 (+0.0056); lost SNppxx (-0.0046), SNpp23 (-0.0045), LgLG3 (-0.0043), LPLC4 (-0.0043), SApp (-0.0041)

Modularity of |W| by superclass: bio 0.3223 -> trained 0.3223; by role: bio 0.0419 -> trained 0.0419

## Leak (integration rate a = sigmoid(leak)) by superclass

| superclass | types | mean a | std |
|---|---|---|---|
| ascending_neuron | 563 | 0.4990 | 0.0022 |
| cb_efferent | 1 | 0.4950 | 0.0000 |
| cb_endocrine | 10 | 0.5028 | 0.0033 |
| cb_intrinsic | 6605 | 0.5000 | 0.0015 |
| cb_motor | 43 | 0.5008 | 0.0039 |
| cb_sensory | 158 | 0.4996 | 0.0024 |
| descending_neuron | 480 | 0.5003 | 0.0043 |
| efferent_ascending | 5 | 0.4991 | 0.0032 |
| efferent_descending | 1 | 0.5033 | 0.0000 |
| ol_intrinsic | 271 | 0.5000 | 0.0001 |
| ol_sensory | 11 | 0.5000 | 0.0002 |
| sensory_ascending | 26 | 0.4986 | 0.0024 |
| sensory_descending | 4 | 0.4972 | 0.0011 |
| visual_centrifugal | 108 | 0.5000 | 0.0009 |
| visual_projection | 346 | 0.4998 | 0.0021 |
| vnc_efferent | 27 | 0.5039 | 0.0044 |
| vnc_endocrine | 4 | 0.5044 | 0.0027 |
| vnc_intrinsic | 2777 | 0.4999 | 0.0020 |
| vnc_motor | 142 | 0.5021 | 0.0039 |
| vnc_sensory | 169 | 0.4997 | 0.0021 |

## Gate usage by source (mean |g| over tokens)

* replay: 0.02538
* code: 0.02538
* bio: 0.02538
* connectome: 0.02538
* fresh: 0.02538

State autocorrelation (cosine of consecutive token states): 0.8718
