# V3.2.2 mandatory contract-17 VAMP search

This result requires every cycle to use a path returned by the VAMP RRT-Connect search tree. Only the two task endpoints, explicit bounds, robot limits and the collision scene are passed to the search. Golden intermediate frames, visibility families and fallback paths are disabled.

## Acceptance

- Three independent seeds: **3/3 passed**.
- Mandatory searches: **45/45 cycles**, **75/75 boxes**.
- Golden-guided/fallback cycles: **0/45**.
- Median 15-cycle planning core: **3.282s**.
- Median/observed maximum cycle: **443.1ms / 494.2ms**.
- VAMP workers solved: **178/180**.
- Dense-FCL worker candidates: **176 passed**, **2 rejected**.
- Direct replay: **28,844 frames + 1,045 edges** passed MoveIt/FCL.
- Safe-turn replay: **25,739 frames + 3,985 edges** passed MoveIt/FCL.

## Per-cycle medians

| Cycle | Solve | Planning | Search | Iterations | Sparse states | Workers | Cost/reference |
|---|---|---:|---:|---:|---:|---:|---:|
| Box 1 + Box 5 | direct_connection | 167.4ms | 2.7ms | 0 | 2 | 12/12 | 0.978 |
| Box 2 + Box 4 | direct_connection | 192.0ms | 2.2ms | 0 | 2 | 12/12 | 1.000 |
| Box 3 | tree_expansion | 268.0ms | 34.9ms | 215 | 10 | 12/12 | 1.050 |
| Box 6 + Box 10 | direct_connection | 156.7ms | 1.4ms | 0 | 2 | 12/12 | 0.702 |
| Box 7 + Box 9 | direct_connection | 158.2ms | 1.7ms | 0 | 2 | 12/12 | 0.983 |
| Box 8 | direct_connection | 228.6ms | 2.1ms | 0 | 2 | 12/12 | 0.950 |
| Box 11 + Box 15 | direct_connection | 172.0ms | 2.6ms | 0 | 2 | 12/12 | 0.208 |
| Box 12 + Box 14 | direct_connection | 171.4ms | 2.4ms | 0 | 2 | 12/12 | 0.273 |
| Box 13 | direct_connection | 199.4ms | 1.7ms | 0 | 2 | 12/12 | 0.910 |
| Box 16 + Box 20 | tree_expansion | 320.8ms | 92.5ms | 823 | 14 | 12/12 | 0.628 |
| Box 17 + Box 19 | direct_connection | 181.2ms | 2.7ms | 0 | 2 | 12/12 | 0.493 |
| Box 18 | direct_connection | 207.5ms | 2.6ms | 0 | 2 | 12/12 | 0.883 |
| Box 21 + Box 25 | tree_expansion | 443.1ms | 205.3ms | 2178 | 19 | 10/12 | 0.723 |
| Box 22 + Box 24 | direct_connection | 187.2ms | 2.4ms | 0 | 2 | 12/12 | 0.994 |
| Box 23 | direct_connection | 192.8ms | 2.6ms | 0 | 2 | 12/12 | 0.822 |

## Interpretation

Twelve cycles are collision-free direct start-to-goal connections, so RRT-Connect correctly returns a two-state path without random expansion. Boxes 3, 16+20 and 21+25 require actual tree expansion in every run. A zero-iteration direct connection is still a planner result: the planner validates the complete edge rather than loading a stored route.

The earlier 0.132s total must remain labeled as the prior-guided fast-path version. The mandatory-search median is slower but has the required path provenance and no hidden fallback.

The historical OMPL result is a 15-D single-worker reference, so the 10.51x search and 2.14x end-to-end ratios are indicative rather than a same-dimensional proof.
