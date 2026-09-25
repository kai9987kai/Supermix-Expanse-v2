# Lab run eedeef4e253dfb2c — X-CNS-001 / window8 / seed 1

3,000 types, 629,875 edges. Config: `{"topology": "real", "sign_mode": "dale", "strength_init": "bio", "edge_param": "dale_softplus", "norm": "frac", "activation": "relu", "state": "window:8", "microsteps": 4, "gate": "static", "subgraph": 3000, "radius": 0.9, "graph_seed": null, "backend": "csr_slab"}`

## Edge drift R = W_trained / W_bio

| min | p01 | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 0.9840 | 0.9971 | 0.9998 | 1.0000 | 1.0000 | 1.0000 | 1.0002 | 1.0029 | 1.0156 |

mean R 1.0000; edges changed > 10%: 0.00%; sign flips: 0

log2 R histogram:

```
 -0.0232 ..  -0.0204 |                                          9
 -0.0204 ..  -0.0175 |                                          35
 -0.0175 ..  -0.0147 |                                          132
 -0.0147 ..  -0.0118 |                                          340
 -0.0118 ..  -0.0090 |                                          837
 -0.0090 ..  -0.0061 |                                          2,015
 -0.0061 ..  -0.0033 |                                          5,724
 -0.0033 ..  -0.0004 | ###                                      39,393
 -0.0004 ..  +0.0024 | ######################################## 568,833
 +0.0024 ..  +0.0053 | #                                        8,252
 +0.0053 ..  +0.0081 |                                          2,647
 +0.0081 ..  +0.0110 |                                          1,064
 +0.0110 ..  +0.0138 |                                          417
 +0.0138 ..  +0.0167 |                                          144
 +0.0167 ..  +0.0195 |                                          25
 +0.0195 ..  +0.0224 |                                          8
```

### By pathway (pre role -> post role)

| pathway | edges | geometric-mean R | std log2 R |
|---|---|---|---|
| central->central | 438,636 | 1.0000 | 0.0009 |
| central->input | 42,609 | 1.0000 | 0.0004 |
| central->output | 37,286 | 1.0001 | 0.0029 |
| input->central | 60,528 | 0.9999 | 0.0018 |
| input->input | 11,790 | 0.9999 | 0.0008 |
| input->output | 4,173 | 1.0006 | 0.0035 |
| output->central | 27,117 | 1.0000 | 0.0010 |
| output->input | 2,271 | 1.0000 | 0.0003 |
| output->output | 5,465 | 1.0002 | 0.0026 |

### Most changed superclass pairs

| pre -> post | edges | geometric-mean R |
|---|---|---|
| vnc_sensory->vnc_motor | 312 | 1.0011 |
| ascending_neuron->vnc_motor | 693 | 1.0010 |
| sensory_ascending->vnc_intrinsic | 128 | 0.9994 |
| ascending_neuron->descending_neuron | 2,030 | 1.0006 |
| visual_projection->descending_neuron | 957 | 1.0002 |
| vnc_sensory->vnc_sensory | 646 | 0.9998 |
| descending_neuron->descending_neuron | 3,493 | 1.0002 |
| vnc_sensory->ascending_neuron | 1,366 | 0.9998 |
| ascending_neuron->cb_intrinsic | 11,818 | 0.9999 |
| descending_neuron->vnc_motor | 1,202 | 1.0001 |
| vnc_intrinsic->vnc_motor | 10,733 | 1.0001 |
| visual_centrifugal->descending_neuron | 835 | 1.0001 |
| ascending_neuron->ascending_neuron | 3,181 | 0.9999 |
| vnc_intrinsic->descending_neuron | 3,779 | 0.9999 |
| cb_sensory->cb_sensory | 751 | 0.9999 |

### Most strengthened edges (synapse count >= median)

| pre | post | synapses | W_bio | W_trained | R |
|---|---|---|---|---|---|
| IN02A054 (vnc_intrinsic) | IN02A044 (vnc_intrinsic) | 689 | -0.19704 | -0.20013 | 1.016 |
| IN08B003 (vnc_intrinsic) | IN06B013 (vnc_intrinsic) | 1316 | -0.10943 | -0.11104 | 1.015 |
| CB4105 (cb_intrinsic) | DNbe003 (descending_neuron) | 136 | +0.02075 | +0.02104 | 1.014 |
| AN05B096 (ascending_neuron) | IN10B006 (vnc_intrinsic) | 240 | +0.09060 | +0.09187 | 1.014 |
| INXXX011 (vnc_intrinsic) | INXXX095 (vnc_intrinsic) | 1318 | +0.21987 | +0.22291 | 1.014 |
| IN10B006 (vnc_intrinsic) | tp1 MN (vnc_motor) | 144 | +0.02737 | +0.02775 | 1.014 |
| SApp (sensory_ascending) | IN02A007 (vnc_intrinsic) | 186 | +0.07191 | +0.07289 | 1.014 |
| INXXX095 (vnc_intrinsic) | tp1 MN (vnc_motor) | 631 | +0.11994 | +0.12157 | 1.014 |
| AN19B019 (ascending_neuron) | PS124 (cb_intrinsic) | 121 | +0.01263 | +0.01280 | 1.013 |
| IN07B086 (vnc_intrinsic) | b1 MN (vnc_motor) | 444 | +0.04998 | +0.05063 | 1.013 |
| IN19B067 (vnc_intrinsic) | tp1 MN (vnc_motor) | 87 | +0.01654 | +0.01675 | 1.013 |
| IN04B006 (vnc_intrinsic) | tp1 MN (vnc_motor) | 346 | +0.06577 | +0.06662 | 1.013 |

### Most weakened edges (synapse count >= median)

| pre | post | synapses | W_bio | W_trained | R |
|---|---|---|---|---|---|
| LT82a (visual_projection) | PS322 (cb_intrinsic) | 167 | +0.02790 | +0.02745 | 0.984 |
| AN03A008 (ascending_neuron) | PS322 (cb_intrinsic) | 85 | +0.01420 | +0.01399 | 0.985 |
| IN06B013 (vnc_intrinsic) | tp1 MN (vnc_motor) | 758 | -0.14408 | -0.14196 | 0.985 |
| IN11B013 (vnc_intrinsic) | tp1 MN (vnc_motor) | 412 | -0.07831 | -0.07718 | 0.986 |
| SNch01 (vnc_sensory) | IN02A044 (vnc_intrinsic) | 852 | +0.24366 | +0.24016 | 0.986 |
| INXXX331 (vnc_intrinsic) | IN02A044 (vnc_intrinsic) | 250 | +0.07150 | +0.07047 | 0.986 |
| AN19B001 (ascending_neuron) | IN06B013 (vnc_intrinsic) | 424 | +0.03526 | +0.03475 | 0.986 |
| IN03B058 (vnc_intrinsic) | tp1 MN (vnc_motor) | 256 | -0.04866 | -0.04797 | 0.986 |
| IN06B066 (vnc_intrinsic) | tp1 MN (vnc_motor) | 280 | -0.05322 | -0.05247 | 0.986 |
| SNxx04 (vnc_sensory) | IN02A044 (vnc_intrinsic) | 50 | +0.01430 | +0.01410 | 0.986 |
| AN04A001 (ascending_neuron) | IN06B013 (vnc_intrinsic) | 584 | +0.04856 | +0.04790 | 0.986 |
| GNG523 (cb_intrinsic) | DNbe003 (descending_neuron) | 188 | -0.02868 | -0.02829 | 0.986 |

### Hubs (change of weighted strength, sum |W|)

* hubs_in_strength: gained KCg-d (+0.0051), IN27X002 (+0.0041), MBON27 (+0.0041), PS196_a (+0.0041), PVLP024 (+0.0040); lost SNch01 (-0.0057), MNad09 (-0.0052), IN11B004 (-0.0049), IN02A030 (-0.0041), SNxx29 (-0.0038)
* hubs_out_strength: gained KCg-m (+0.0106), KCg-d (+0.0087), LC10d (+0.0065), lLN2F_b (+0.0038), LC10c-2 (+0.0033); lost SNch01 (-0.0114), LC18 (-0.0106), AN19A018 (-0.0100), LPC1 (-0.0078), SApp (-0.0075)

Modularity of |W| by superclass: bio 0.3563 -> trained 0.3563; by role: bio 0.0193 -> trained 0.0193

## Leak (integration rate a = sigmoid(leak)) by superclass

| superclass | types | mean a | std |
|---|---|---|---|
| ascending_neuron | 94 | 0.5024 | 0.0059 |
| cb_endocrine | 1 | 0.5190 | 0.0000 |
| cb_intrinsic | 1523 | 0.5014 | 0.0042 |
| cb_motor | 2 | 0.4993 | 0.0060 |
| cb_sensory | 60 | 0.5010 | 0.0023 |
| descending_neuron | 103 | 0.5011 | 0.0075 |
| ol_intrinsic | 233 | 0.5000 | 0.0004 |
| ol_sensory | 9 | 0.4997 | 0.0004 |
| sensory_ascending | 2 | 0.4955 | 0.0005 |
| visual_centrifugal | 74 | 0.5006 | 0.0026 |
| visual_projection | 104 | 0.5011 | 0.0057 |
| vnc_efferent | 1 | 0.5143 | 0.0000 |
| vnc_endocrine | 1 | 0.5159 | 0.0000 |
| vnc_intrinsic | 662 | 0.5006 | 0.0048 |
| vnc_motor | 74 | 0.5001 | 0.0078 |
| vnc_sensory | 57 | 0.5026 | 0.0054 |

## Gate usage by source (mean |g| over tokens)

* replay: 0.05100
* code: 0.05100
* bio: 0.05100
* connectome: 0.05100
* fresh: 0.05100
* temporal: 0.05100

State autocorrelation (cosine of consecutive token states): 0.9461
