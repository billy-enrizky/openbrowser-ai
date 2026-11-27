# Seven-interface N=5 scoreboard, 2026-11-27

<!-- toc:start -->
## Contents

- [1. Scope](#1-scope)
- [2. Time](#2-time)
  - [2.1. Cold](#21-cold)
  - [2.2. Warm](#22-warm)
- [3. Memory](#3-memory)
- [4. Retained loss queue](#4-retained-loss-queue)

<!-- toc:end -->

## 1. Scope

Cold: 175 exact latency observations and 175 valid resource observations. Warm: 140 exact latency observations and 140 valid resource observations. Each cell has N=5. Arms rotate between samples. Chrome for Testing 159.0.8154.6, revision 1255, is pinned on one machine. OpenBrowser app digest starts `847bbf45`.

Time and memory: lower is better ↓. Native task time excludes external metadata validation and teardown. RSS is the sampled peak native child-tree RSS, with the benchmark worker, controller, and fixture server outside the score. Summed RSS can count shared pages more than once. Resource CPU is a sampled lower bound. Resource-instrumented wall time supplies no latency claim.

N=5 p95 equals the maximum observation under the harness percentile formula. Every observation remains in the raw JSONL. Cold and warm use separate harness receipts and the same application bytes.

| Lane | Median time wins / losses | p95 time wins / losses | Median peak RSS wins / losses |
| --- | ---: | ---: | ---: |
| Cold | 13 / 7 | 12 / 8 | 20 / 0 |
| Warm | 16 / 0 | 16 / 0 | 16 / 0 |

Counts compare each OpenBrowser interface against both Playwright interfaces on every task. A loss or tie stays in the loss count.

## 2. Time

Each value is median / p95 in milliseconds. Lower is better ↓.

### 2.1. Cold

| Task | OpenBrowser MCP | OpenBrowser CLI | Playwright MCP | Playwright CLI |
| --- | ---: | ---: | ---: | ---: |
| wizard | 699.861 / 1146.867 | 795.955 / 824.244 | 941.537 / 949.036 | 1732.262 / 1853.461 |
| modal | 823.093 / 841.969 | 945.112 / 970.386 | 850.733 / 934.007 | 1709.790 / 1727.923 |
| download | 978.479 / 1036.389 | 1075.754 / 1105.994 | 938.340 / 951.259 | 1738.421 / 1783.139 |
| tabs | 722.224 / 768.955 | 812.648 / 814.743 | 669.406 / 691.612 | 1491.717 / 1561.706 |
| auth | 1140.787 / 1188.648 | 1365.922 / 1452.703 | 1038.182 / 1078.800 | 2889.772 / 2955.748 |

### 2.2. Warm

| Task | OpenBrowser MCP | OpenBrowser CLI | Playwright MCP | Playwright CLI |
| --- | ---: | ---: | ---: | ---: |
| wizard | 54.394 / 85.800 | 93.174 / 109.643 | 359.009 / 380.117 | 561.216 / 565.046 |
| modal | 204.733 / 216.366 | 237.042 / 249.272 | 282.960 / 288.477 | 479.470 / 498.992 |
| download | 342.371 / 359.854 | 364.725 / 371.380 | 371.945 / 378.389 | 570.364 / 587.300 |
| tabs | 78.586 / 84.018 | 114.917 / 124.138 | 128.525 / 129.309 | 309.912 / 313.608 |

## 3. Memory

Median sampled peak native child-tree RSS, MiB. Lower is better ↓.

| Lane | Task | OpenBrowser MCP | OpenBrowser CLI | Playwright MCP | Playwright CLI |
| --- | --- | ---: | ---: | ---: | ---: |
| Cold | wizard | 720.4 | 746.5 | 806.5 | 918.3 |
| Cold | modal | 718.9 | 750.1 | 788.8 | 901.7 |
| Cold | download | 709.9 | 737.5 | 772.7 | 865.3 |
| Cold | tabs | 723.5 | 752.5 | 792.3 | 897.7 |
| Cold | auth | 727.4 | 749.1 | 797.7 | 902.0 |
| Warm | wizard | 662.1 | 667.6 | 740.0 | 847.9 |
| Warm | modal | 656.2 | 662.4 | 730.8 | 850.2 |
| Warm | download | 652.0 | 659.1 | 713.9 | 841.7 |
| Warm | tabs | 658.1 | 664.4 | 736.8 | 845.0 |

## 4. Retained loss queue

| Lane | Task | OpenBrowser | Other interface | Metric | OpenBrowser | Playwright |
| --- | --- | --- | --- | --- | ---: | ---: |
| cold | wizard | OpenBrowser MCP | Playwright MCP | p95_time | 1146.867 | 949.036 |
| cold | modal | OpenBrowser CLI | Playwright MCP | median_time | 945.112 | 850.733 |
| cold | modal | OpenBrowser CLI | Playwright MCP | p95_time | 970.386 | 934.007 |
| cold | download | OpenBrowser MCP | Playwright MCP | median_time | 978.479 | 938.340 |
| cold | download | OpenBrowser MCP | Playwright MCP | p95_time | 1036.389 | 951.259 |
| cold | download | OpenBrowser CLI | Playwright MCP | median_time | 1075.754 | 938.340 |
| cold | download | OpenBrowser CLI | Playwright MCP | p95_time | 1105.994 | 951.259 |
| cold | tabs | OpenBrowser MCP | Playwright MCP | median_time | 722.224 | 669.406 |
| cold | tabs | OpenBrowser MCP | Playwright MCP | p95_time | 768.955 | 691.612 |
| cold | tabs | OpenBrowser CLI | Playwright MCP | median_time | 812.648 | 669.406 |
| cold | tabs | OpenBrowser CLI | Playwright MCP | p95_time | 814.743 | 691.612 |
| cold | auth | OpenBrowser MCP | Playwright MCP | median_time | 1140.787 | 1038.182 |
| cold | auth | OpenBrowser MCP | Playwright MCP | p95_time | 1188.648 | 1078.800 |
| cold | auth | OpenBrowser CLI | Playwright MCP | median_time | 1365.922 | 1038.182 |
| cold | auth | OpenBrowser CLI | Playwright MCP | p95_time | 1452.703 | 1078.800 |
