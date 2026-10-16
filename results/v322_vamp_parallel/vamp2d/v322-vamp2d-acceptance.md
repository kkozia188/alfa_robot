# V3.2.2 2-D VAMP acceptance

- Same-input planning core: **26.150s (14-D) -> 14.295s (VAMP)**, **1.83x**.
- Fair 2-D lattice -> VAMP: **14.557s -> 14.295s**, **1.018x**.
- Latest direct-pull VAMP core: **3.428s total**, maximum **1.143s**.
- Pure VAMP wall time: **5.648ms** across 15 cycles; workers solved **60/60**.
- Winning candidates passing full FCL: **15/15**.
- Final safe-turn validation: **25,731 frames + 4,011 edge samples passed**.
- Rerun blue bars: **25/25 under 3s**, maximum **1.143s**.

The visibility graph remains the selected path because its joint travel is lower. VAMP is genuinely executed as a four-worker AVX2 planner and is retained as the validated concurrent fallback. The Rerun layout is unchanged; only the orbit target is fixed at the warehouse/robot world origin.
