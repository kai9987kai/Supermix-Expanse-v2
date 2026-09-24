# Connectome Lab core benchmark

8 x 128 tokens, hidden 320, 8 threads, fp32, CPU (torch 2.11.0+cpu). Core alone; seconds per call.

| graph | edges | backend | edges learned | state | forward s | backward s | step s | process peak RSS MB (running max) |
|---|---|---|---|---|---|---|---|---|
| full | 3,830,931 | csr_slab | fixed | reset | 0.866 | 0.879 | 1.745 | 1345.2 |
| full | 3,830,931 | csr_slab | fixed | persist | 2.277 | 3.557 | 5.834 | 1546.7 |
| full | 3,830,931 | csr_slab | dale_softplus | reset | 1.095 | 4.764 | 5.859 | 1706.0 |
| full | 3,830,931 | csr_slab | dale_softplus | persist | 2.267 | 7.069 | 9.337 | 1862.7 |
| full | 3,830,931 | torch_sparse | fixed | reset | 26.452 | 26.83 | 53.281 | 1862.7 |
| full | 3,830,931 | torch_sparse | fixed | persist | per-token scan on the full graph: csr_slab only | | |  |
| full | 3,830,931 | torch_sparse | dale_softplus | reset | 28.043 | 38.014 | 66.057 | 2429.8 |
| full | 3,830,931 | torch_sparse | dale_softplus | persist | per-token scan on the full graph: csr_slab only | | |  |
| full | 3,830,931 | dense | fixed | reset | dense n x n too large | | |  |
| full | 3,830,931 | dense | fixed | persist | dense n x n too large | | |  |
| full | 3,830,931 | dense | dale_softplus | reset | dense n x n too large | | |  |
| full | 3,830,931 | dense | dale_softplus | persist | dense n x n too large | | |  |
| 3000 | 629,875 | csr_slab | fixed | reset | 0.194 | 0.234 | 0.428 | 2429.8 |
| 3000 | 629,875 | csr_slab | fixed | persist | 0.933 | 1.925 | 2.857 | 2429.8 |
| 3000 | 629,875 | csr_slab | dale_softplus | reset | 0.181 | 0.796 | 0.977 | 2429.8 |
| 3000 | 629,875 | csr_slab | dale_softplus | persist | 0.969 | 3.219 | 4.188 | 2429.8 |
| 3000 | 629,875 | torch_sparse | fixed | reset | 4.336 | 4.84 | 9.176 | 2429.8 |
| 3000 | 629,875 | torch_sparse | fixed | persist | 28.384 | 27.746 | 56.129 | 2429.8 |
| 3000 | 629,875 | torch_sparse | dale_softplus | reset | 3.56 | 3.834 | 7.395 | 2429.8 |
| 3000 | 629,875 | torch_sparse | dale_softplus | persist | 23.982 | 82.069 | 106.051 | 2429.8 |
| 3000 | 629,875 | dense | fixed | reset | 0.423 | 0.529 | 0.953 | 2429.8 |
| 3000 | 629,875 | dense | fixed | persist | 5.377 | 8.156 | 13.533 | 2429.8 |
| 3000 | 629,875 | dense | dale_softplus | reset | 0.338 | 0.972 | 1.31 | 2429.8 |
| 3000 | 629,875 | dense | dale_softplus | persist | 8.252 | 11.612 | 19.863 | 2429.8 |
| 1500 | 204,500 | csr_slab | fixed | reset | 0.068 | 0.062 | 0.131 | 2429.8 |
| 1500 | 204,500 | csr_slab | fixed | persist | 0.496 | 0.989 | 1.485 | 2429.8 |
| 1500 | 204,500 | csr_slab | dale_softplus | reset | 0.112 | 0.29 | 0.402 | 2429.8 |
| 1500 | 204,500 | csr_slab | dale_softplus | persist | 0.685 | 1.313 | 1.998 | 2429.8 |
| 1500 | 204,500 | torch_sparse | fixed | reset | 1.493 | 1.346 | 2.839 | 2429.8 |
| 1500 | 204,500 | torch_sparse | fixed | persist | 12.651 | 14.45 | 27.101 | 2429.8 |
| 1500 | 204,500 | torch_sparse | dale_softplus | reset | 1.911 | 2.105 | 4.016 | 2429.8 |
| 1500 | 204,500 | torch_sparse | dale_softplus | persist | 9.024 | 39.129 | 48.152 | 2429.8 |
| 1500 | 204,500 | dense | fixed | reset | 0.239 | 0.258 | 0.497 | 2429.8 |
| 1500 | 204,500 | dense | fixed | persist | 2.175 | 3.159 | 5.334 | 2429.8 |
| 1500 | 204,500 | dense | dale_softplus | reset | 0.244 | 0.424 | 0.669 | 2429.8 |
| 1500 | 204,500 | dense | dale_softplus | persist | 2.065 | 4.472 | 6.538 | 2429.8 |
