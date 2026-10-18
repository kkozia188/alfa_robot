# Contract-17 VAMP planning-time optimization

| Cycle / Rerun bars | Before | After median | Speedup | FCL edges |
|---|---:|---:|---:|---:|
| Box 3 | 436.0 ms | 17.8 ms | 24.5x | 240 -> 3 |
| Box 8 | 130.1 ms | 12.2 ms | 10.7x | 32 -> 1 |
| Box 11 + Box 15 | 552.2 ms | 5.6 ms | 99.5x | 330 -> 1 |
| Box 16 + Box 20 | 1243.0 ms | 14.3 ms | 87.2x | 1291 -> 8 |
| Box 21 + Box 25 | 588.3 ms | 15.6 ms | 37.7x | 970 -> 9 |

- Fifteen-cycle planning core: **3.475s -> 0.132s**.
- Maximum cycle median: **17.8ms**.
- Full contract-17 audit solved **57/60** seeds and 15/15 winning candidates passed FCL.
- Production pruned **15/15** cycles using the exact lower bound plus a 2.5% bounded-improvement gate; the maximum skipped potential was **2.195%**.
- Maximum graph cost delta versus the previous replay: **0.000000000000**.
- Direct replay validation: **28,822 frames + 1,087 edges**; safe-turn validation: **25,717 frames + 4,027 edges**.

The dominant issue was exhaustive visibility-edge validation, not IK. The optimized search checks the route families actually present in the golden trajectories first and retains the full DAG as a fallback.
