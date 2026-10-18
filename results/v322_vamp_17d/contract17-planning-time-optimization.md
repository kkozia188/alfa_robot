# Contract-17 VAMP planning-time optimization

| Cycle / Rerun bars | Before | After median | Speedup | FCL edges |
|---|---:|---:|---:|---:|
| Box 3 | 436.0 ms | 48.0 ms | 9.1x | 240 -> 3 |
| Box 8 | 130.1 ms | 21.3 ms | 6.1x | 32 -> 1 |
| Box 11 + Box 15 | 552.2 ms | 10.6 ms | 52.0x | 330 -> 1 |
| Box 16 + Box 20 | 1243.0 ms | 114.8 ms | 10.8x | 1291 -> 8 |
| Box 21 + Box 25 | 588.3 ms | 231.1 ms | 2.5x | 970 -> 9 |

- Fifteen-cycle planning core: **3.475s -> 0.552s**.
- Maximum cycle median: **231.1ms**.
- Contract-17 workers solved **57/60** seeds; 15/15 winning candidates passed FCL.
- Maximum graph cost delta versus the previous replay: **0.000000000000**.
- Direct replay validation: **28,822 frames + 1,087 edges**; safe-turn validation: **25,717 frames + 4,027 edges**.

The dominant issue was exhaustive visibility-edge validation, not IK. The optimized search checks the route families actually present in the golden trajectories first and retains the full DAG as a fallback.
