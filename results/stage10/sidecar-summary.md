## Stage 10 Real Sidecar Calibration Matrix

| run | model | load | real high/low p95 | real corr. | sync ratio | p95 ms | p99 ms | tok/s | sim high/low |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vllm-0p5b-baseline | Qwen/Qwen2.5-0.5B-Instruct | c=1, short, new=32 | 1.0047 | 0.0464 | 1.0008 | 152.7 | 154.4 | 211.4 | 1.7179 |
| vllm-0p5b-highcon | Qwen/Qwen2.5-0.5B-Instruct | c=8, mixed, new=96 | 1.0020 | 0.7603 | 1.0002 | 602.7 | 603.9 | 1385.7 | 1.7191 |
| vllm-1p5b-highcon | Qwen/Qwen2.5-1.5B-Instruct | c=8, long, new=128 | 1.0034 | 0.7726 | 1.0008 | 1185.9 | 1187.8 | 1057.3 | 1.7204 |
| vllm-1p5b-medium | Qwen/Qwen2.5-1.5B-Instruct | c=4, mixed, new=96 | 1.0031 | 0.0761 | 1.0006 | 686.1 | 687.1 | 594.3 | 1.7175 |
| vllm-1p5b-pressure-heavy | Qwen/Qwen2.5-1.5B-Instruct | c=8, mixed, new=128 | 1.5461 | 0.8979 | 0.9998 | 1158.2 | 1160.7 | 1113.4 | 1.7093 |
| vllm-1p5b-sync-heavy | Qwen/Qwen2.5-1.5B-Instruct | c=8, mixed, new=128 | 1.0001 | 0.0164 | 1.5560 | 1156.6 | 1158.3 | 1280.6 | 1.7115 |
| vllm-3b-highcon | Qwen/Qwen2.5-3B-Instruct | c=8, long, new=128 | 1.0034 | 0.7808 | 1.0006 | 1658.7 | 1660.7 | 798.3 | 1.7163 |
| vllm-7b-high | Qwen/Qwen2.5-7B-Instruct | c=4, long, new=128 | 1.0046 | 0.7745 | 1.0005 | 1710.9 | 1713.7 | 440.2 | 1.7088 |
