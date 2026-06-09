# Claim Hierarchy

The paper is protocol-first: each evidence tier supports a specific claim and does not automatically license the next one.

| Tier | Evidence required | This artifact contains | Valid claim |
| --- | --- | --- | --- |
| Stress harness | Controlled pressure replay with fixed and deferral baselines | Configs, simulator, training/controller code, aggregate table inputs | Synchronization scheduling can matter in a repeatable benchmark |
| Ordinary sidecar | Real p95 effect-size transfer | Stage 10 vLLM calibration summaries | Ordinary vLLM did not support a production SLO claim from the harness alone |
| Targeted sidecar | Mechanism-specific transfer | Pressure-heavy and sync-heavy calibration summaries | Pressure and sync-window mechanisms must be calibrated separately |
| Steady replay | Matched policy traces and matched deferral | Stage 11 policy replay summaries | WA-DiLoCo beats fixed/gate baselines in sync-heavy replay but ties matched random |
| Bursty offline replay | No-sync load match and random-draw envelope | Combined Stage 12/13 aggregate summary | Calibrated serving-cost signal has value in replay |
| Online controller | Decision-time signal only | Stage 13 online calibrated-WA summaries | Online calibrated-WA is directional but remains tied with matched random |
| Fabric smoke | Real collective operations on multi-node fabric | Stage 14 MI355X/RCCL summaries | Outer-sync-sized collectives run on real fabric; this is not full multi-node policy replay |

Use these boundaries when reusing the code or citing the artifact. In particular, the repo should not be described as showing a universal online-controller win over matched random deferral.
