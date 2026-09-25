# Lab run e0deb7e7e23d0efc — H-CNS-006 / dynamic_shuffled / seed 4

3,000 types, 629,875 edges. Config: `{"topology": "shuffled", "sign_mode": "dale", "strength_init": "bio", "edge_param": "dale_softplus", "norm": "frac", "activation": "relu", "state": "reset", "microsteps": 4, "gate": "dynamic", "subgraph": 3000, "radius": 0.9, "graph_seed": null, "backend": "csr_slab"}`

## Edge drift R = W_trained / W_bio

| min | p01 | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 0.9870 | 0.9961 | 0.9995 | 1.0000 | 1.0000 | 1.0000 | 1.0004 | 1.0039 | 1.0127 |

mean R 1.0000; edges changed > 10%: 0.00%; sign flips: 0

log2 R histogram:

```
 -0.0188 ..  -0.0165 |                                          10
 -0.0165 ..  -0.0142 |                                          52
 -0.0142 ..  -0.0119 |                                          203
 -0.0119 ..  -0.0096 |                                          674
 -0.0096 ..  -0.0073 |                                          2,043
 -0.0073 ..  -0.0050 |                                          5,265
 -0.0050 ..  -0.0026 | #                                        13,812
 -0.0026 ..  -0.0003 | #####                                    66,969
 -0.0003 ..  +0.0020 | ######################################## 511,108
 +0.0020 ..  +0.0043 | #                                        18,762
 +0.0043 ..  +0.0066 | #                                        7,077
 +0.0066 ..  +0.0089 |                                          2,683
 +0.0089 ..  +0.0112 |                                          898
 +0.0112 ..  +0.0135 |                                          260
 +0.0135 ..  +0.0158 |                                          50
 +0.0158 ..  +0.0182 |                                          9
```

### By pathway (pre role -> post role)

| pathway | edges | geometric-mean R | std log2 R |
|---|---|---|---|
| central->central | 438,636 | 1.0000 | 0.0009 |
| central->input | 42,609 | 1.0000 | 0.0004 |
| central->output | 37,286 | 1.0000 | 0.0030 |
| input->central | 60,528 | 1.0000 | 0.0031 |
| input->input | 11,790 | 1.0000 | 0.0017 |
| input->output | 4,173 | 1.0005 | 0.0039 |
| output->central | 27,117 | 1.0000 | 0.0006 |
| output->input | 2,271 | 1.0000 | 0.0004 |
| output->output | 5,465 | 1.0001 | 0.0028 |

### Most changed superclass pairs

| pre -> post | edges | geometric-mean R |
|---|---|---|
| ascending_neuron->vnc_motor | 487 | 1.0011 |
| visual_projection->vnc_motor | 374 | 1.0010 |
| vnc_sensory->vnc_motor | 131 | 1.0007 |
| cb_sensory->visual_centrifugal | 152 | 0.9996 |
| visual_projection->descending_neuron | 1,186 | 1.0003 |
| ascending_neuron->descending_neuron | 1,385 | 1.0003 |
| ol_intrinsic->vnc_motor | 990 | 0.9997 |
| ol_sensory->cb_intrinsic | 278 | 1.0003 |
| vnc_sensory->descending_neuron | 353 | 1.0002 |
| cb_sensory->ascending_neuron | 260 | 0.9998 |
| vnc_motor->descending_neuron | 428 | 1.0002 |
| ol_intrinsic->descending_neuron | 2,366 | 0.9998 |
| ascending_neuron->visual_centrifugal | 1,389 | 0.9998 |
| visual_projection->visual_centrifugal | 1,273 | 0.9999 |
| vnc_motor->vnc_motor | 253 | 1.0001 |

### Most strengthened edges (synapse count >= median)

| pre | post | synapses | W_bio | W_trained | R |
|---|---|---|---|---|---|
| AN04A001 (ascending_neuron) | IN23B006 (vnc_intrinsic) | 2582 | +0.24384 | +0.24693 | 1.013 |
| LC31b (visual_projection) | IN23B020 (vnc_intrinsic) | 543 | +0.03796 | +0.03844 | 1.013 |
| LC31b (visual_projection) | PLP211 (cb_intrinsic) | 178 | +0.02632 | +0.02665 | 1.012 |
| AN17A015 (ascending_neuron) | AVLP575 (cb_intrinsic) | 54 | +0.00932 | +0.00943 | 1.012 |
| LC31b (visual_projection) | PLP231 (cb_intrinsic) | 515 | +0.12599 | +0.12748 | 1.012 |
| LC31b (visual_projection) | IN17A023 (vnc_intrinsic) | 84 | +0.01089 | +0.01102 | 1.012 |
| AN04A001 (ascending_neuron) | WED210 (cb_intrinsic) | 25 | +0.00638 | +0.00645 | 1.011 |
| LC31b (visual_projection) | CB1072 (cb_intrinsic) | 338 | +0.03047 | +0.03082 | 1.011 |
| AN04B001 (ascending_neuron) | PS047_b (cb_intrinsic) | 451 | +0.06493 | +0.06565 | 1.011 |
| MeVP1 (visual_projection) | WED210 (cb_intrinsic) | 84 | +0.02143 | +0.02167 | 1.011 |
| LC6 (visual_projection) | IN04B004 (vnc_intrinsic) | 8 | +0.00269 | +0.00272 | 1.011 |
| AN17A015 (ascending_neuron) | IN09B038 (vnc_intrinsic) | 84 | +0.01282 | +0.01296 | 1.011 |

### Most weakened edges (synapse count >= median)

| pre | post | synapses | W_bio | W_trained | R |
|---|---|---|---|---|---|
| LC31b (visual_projection) | IN13A005 (vnc_intrinsic) | 716 | +0.03401 | +0.03357 | 0.987 |
| LC31b (visual_projection) | INXXX004 (vnc_intrinsic) | 340 | +0.04476 | +0.04419 | 0.987 |
| LC31b (visual_projection) | PLP256 (cb_intrinsic) | 270 | +0.01589 | +0.01569 | 0.987 |
| AN09B002 (ascending_neuron) | CB0743 (cb_intrinsic) | 277 | +0.02729 | +0.02696 | 0.988 |
| AN04A001 (ascending_neuron) | IN13A006 (vnc_intrinsic) | 906 | +0.05647 | +0.05580 | 0.988 |
| LC6 (visual_projection) | PVLP100 (cb_intrinsic) | 118 | +0.00535 | +0.00528 | 0.988 |
| LLPC3 (visual_projection) | PVLP100 (cb_intrinsic) | 99 | +0.00449 | +0.00443 | 0.988 |
| ANXXX145 (ascending_neuron) | IN19A022 (vnc_intrinsic) | 86 | +0.00404 | +0.00400 | 0.989 |
| LC9 (visual_projection) | IN19A022 (vnc_intrinsic) | 1160 | +0.05455 | +0.05393 | 0.989 |
| LC31b (visual_projection) | PLP060 (cb_intrinsic) | 24 | +0.00294 | +0.00291 | 0.989 |
| LoVP54 (visual_projection) | IN12B027 (vnc_intrinsic) | 342 | +0.05941 | +0.05874 | 0.989 |
| SNta35 (vnc_sensory) | IN09A003 (vnc_intrinsic) | 1670 | +0.06927 | +0.06849 | 0.989 |

### Hubs (change of weighted strength, sum |W|)

* hubs_in_strength: gained PVLP025 (+0.0070), PLP231 (+0.0052), AVLP548_g1 (+0.0042), PLP248 (+0.0042), DP1m_adPN (+0.0041); lost MBON30 (-0.0058), LPi21 (-0.0054), AVLP464 (-0.0053), PVLP072 (-0.0049), ExR6 (-0.0048)
* hubs_out_strength: gained aMe9 (+0.0081), LC4 (+0.0066), LPLC2 (+0.0057), LgLG3 (+0.0052), AN04B003 (+0.0047); lost LLPC2 (-0.0059), AN04B001 (-0.0057), PVLP072 (-0.0055), AN09B002 (-0.0053), PS230 (-0.0051)

Modularity of |W| by superclass: bio 0.0291 -> trained 0.0291; by role: bio 0.0193 -> trained 0.0193

## Leak (integration rate a = sigmoid(leak)) by superclass

| superclass | types | mean a | std |
|---|---|---|---|
| ascending_neuron | 94 | 0.4993 | 0.0035 |
| cb_endocrine | 1 | 0.4897 | 0.0000 |
| cb_intrinsic | 1523 | 0.5000 | 0.0044 |
| cb_motor | 2 | 0.5040 | 0.0025 |
| cb_sensory | 60 | 0.5011 | 0.0036 |
| descending_neuron | 103 | 0.5011 | 0.0042 |
| ol_intrinsic | 233 | 0.4985 | 0.0042 |
| ol_sensory | 9 | 0.5011 | 0.0026 |
| sensory_ascending | 2 | 0.4983 | 0.0016 |
| visual_centrifugal | 74 | 0.4982 | 0.0051 |
| visual_projection | 104 | 0.5002 | 0.0037 |
| vnc_efferent | 1 | 0.5064 | 0.0000 |
| vnc_endocrine | 1 | 0.5008 | 0.0000 |
| vnc_intrinsic | 662 | 0.4994 | 0.0052 |
| vnc_motor | 74 | 0.5007 | 0.0035 |
| vnc_sensory | 57 | 0.4999 | 0.0034 |

## Gate usage by source (mean |g| over tokens)

* replay: 0.54850
* code: 0.57414
* bio: 0.57394
* connectome: 0.57311
* fresh: 0.55154
* temporal: 0.64305

State autocorrelation (cosine of consecutive token states): 0.8307
