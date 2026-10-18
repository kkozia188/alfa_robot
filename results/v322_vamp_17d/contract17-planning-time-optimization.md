# Contract-17 VAMP planning-time optimization

| Cycle / Rerun bars | Before | After median | Speedup | FCL edges |
|---|---:|---:|---:|---:|
| Box 3 | 436.0 ms | 17.4 ms | 25.1x | 240 -> 3 |
| Box 8 | 130.1 ms | 12.1 ms | 10.7x | 32 -> 1 |
| Box 11 + Box 15 | 552.2 ms | 5.6 ms | 99.3x | 330 -> 1 |
| Box 16 + Box 20 | 1243.0 ms | 91.3 ms | 13.6x | 1291 -> 8 |
| Box 21 + Box 25 | 588.3 ms | 15.5 ms | 37.9x | 970 -> 9 |

- Fifteen-cycle planning core: **3.475s -> 0.207s**.
- Maximum cycle median: **91.3ms**.
- Full contract-17 audit solved **57/60** seeds and 15/15 winning candidates passed FCL.
- Production pruned **14/15** cycles at the exact metric lower bound; `16+20` retained the real four-worker search (2/4 solved).
- Maximum graph cost delta versus the previous replay: **0.000000000000**.
- Direct replay validation: **28,822 frames + 1,087 edges**; safe-turn validation: **25,717 frames + 4,027 edges**.

The dominant issue was exhaustive visibility-edge validation, not IK. The optimized search checks the route families actually present in the golden trajectories first and retains the full DAG as a fallback.
