# Production-safe SIMD sphere refinement plan

## Goal

Replace the current approximate robot-only sphere kernel with conservative,
payload-aware kernels that can execute a complete 25-box task without relying
on an unsafe collision label. Removing the final FCL gate is allowed only after
all required state and motion audits have zero false negatives.

## Seed generation

1. Split every collision mesh into connected components and preserve thin-shell
   components instead of using signed-volume tests.
2. Sample triangle centroids proportional to area, then add deterministic seeds
   at high-curvature vertices, boundary edges, joint-adjacent geometry, and the
   farthest uncovered surface point.
3. Assign triangles to their nearest seed. Set each sphere radius to the maximum
   distance from the seed to every assigned triangle vertex plus a safety margin.
   This is an outer cover, not an inscribed fit.
4. Repeat farthest-point insertion until the maximum uncovered distance and
   volume inflation satisfy their per-link limits.
5. Merge spheres only when the merged sphere still covers every source triangle
   and does not violate the per-link over-coverage limit.

## Payload kernels

Generate and benchmark separate collision variants:

- empty robot;
- left/front payload;
- right/front payload;
- left/top payload;
- right/top payload;
- dual/front payloads;
- dual/top or mixed payloads required by the 5x5 task.

Payload spheres are expressed in Tool0 coordinates and transformed by the same
generated FK kernel as the end-effector link. The attachment transform and box
dimensions are part of the kernel manifest and checksum.

## Counterexample-guided refinement

For every state where FCL reports collision and the sphere kernel reports clear:

1. save joint state, colliding object pair, contact point, and penetration depth;
2. map the contact witness into the responsible link frame;
3. insert or enlarge the nearest seed sphere with the minimum margin needed to
   cover the witness;
4. regenerate the kernel and rerun the complete accumulated counterexample set;
5. stop only after zero false negatives on all validation suites.

False positives are tracked separately. They reduce planning completeness and
must be minimized after the zero-false-negative constraint is met.

## Required audits

- at least 100,000 deterministic full-domain random states;
- focused states around all 25 task IK endpoints;
- every state and 1 degree / 1 cm edge sample from OMPL and VAMP task paths;
- adversarial narrow-passage samples around box gaps, warehouse panels, wrist
  crossings, and the 8 degree carried-box tilt boundary;
- all empty, single-payload, and dual-payload kernel variants;
- three planning seeds per box and backend.

Acceptance requires zero sphere false negatives in every suite. A random-state
pass alone is insufficient.

## Hybrid rollout

Before removing FCL entirely, use the conservative SIMD kernel as broad phase:

- reject sphere-colliding states immediately;
- accept states with clearance above a certified margin;
- run FCL only for boundary states and every returned path edge;
- record the fraction of states sent to FCL and end-to-end search time.

The hybrid version is accepted when it retains zero final FCL rejections and
provides a measurable search-time reduction. Pure SIMD replaces the final FCL
gate only after the stronger audits above pass.

## Final task acceptance

- 25/25 boxes complete precontact, contact, attach, retreat, transport, release,
  and return-to-home using newly searched RRT-Connect paths;
- no historical path, shortcut, Cartesian segment, or fallback;
- all actual path edges independently pass MoveIt/FCL;
- the continuous English Rerun shows attached payload motion and continuous base
  station transitions without teleportation;
- the report includes absolute OMPL/VAMP times, winning-attempt time, cumulative
  time to first solution, speedup, success rate, and all rejected attempts.
