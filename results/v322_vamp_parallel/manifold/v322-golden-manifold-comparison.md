# Golden-manifold search comparison

- State dimension: **14 -> 2**.
- Exact candidates: **2/15 -> 15/15**.
- Total planning core: **26.150s -> 14.079s** (**1.86x**).
- Maximum cycle core: **2.502s -> 1.608s**.
- Final operation frame hashes: **16/16 identical**.
- MoveIt/FCL: **43,366 frames + 5,901 edge samples passed**.

The visibility graph remains the selected path because all lattice candidates have slightly greater joint travel. The two-dimensional lattice is retained as the deterministic fallback and as the future VAMP batch-validation boundary. No VAMP speedup is claimed until the V3 sphere kernel is validated.
