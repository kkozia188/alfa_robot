# V3.2.2 full online Pose-to-VAMP experiment

This experiment does not load historical replay joint endpoints, intermediate
frames, waypoints, shortcut paths, or fallback paths. Arm goals are generated
online from the current state, target box/TCP pose, robot limits, and the
current collision scene.

The dimensional contract is reported explicitly:

- grasp, contact, loaded retreat, and chest carry: 15D (`updown` + two 7-axis arms);
- loaded base delivery, joint dual-arm stow, and empty base return: 17D
  (`base x/y/yaw` + two 7-axis arms, with `updown` projected from endpoints).

Consequently, this is full online VAMP functional evidence, but it is not yet
the final same-17D OMPL-versus-VAMP latency comparison. Dual pickup cycles are
formed by synchronizing two independently searched online arm trees and are
then checked as a combined trajectory by MoveIt/FCL. The safe-exit stow is a
single joint dual-arm VAMP tree.

Authoritative artifacts:

- `full-online-vamp-safe-turn.rrd`
- `full-online-vamp-replay.json`
- `full-online-vamp-validation.json`
- `full-online-vamp-safe-turn.json`
- `full-online-vamp-report.json`
- `base-report.json`

Acceptance result: 25/25 boxes, 15/15 cycles, 18,688 frames and 2,466 dense
edge samples passed MoveIt/FCL. The Rerun SHA-256 is
`3914af55a9acf7158704daa61a4734144bf2fd11503f0b8f2e0a7ef63751a0aa`.

