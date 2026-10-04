# Frozen V3.2.2 cuRobo collision model

This directory is the loadable collision-model snapshot for
`robot_description@6bb184b`.

- Robot model: `alfa_v322_suction_final.yml` (`260` spheres, `18` links).
- BaseLink: `33` retained MorphIt spheres from the accepted parameter set.
- Attached box: `box_morphit_final.json` (`24` MorphIt spheres).
- Tool0: both sides are `0.151 m` forward from link7.
- Default attached-box centre: another `0.15 m` along Tool0 local +Z.
- Source parameters and counts: `manifest.json`.
- Integrity hashes: `SHA256SUMS`.

The YAML contains absolute paths to the metre-scaled URDF in this directory,
so moving the directory requires updating `asset_root_path` and `urdf_path`.
The snapshot freezes collision geometry; it does not remove the requirement for
exact mesh and box OBB validation before execution.
