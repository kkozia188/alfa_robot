#!/usr/bin/env bash
set -o pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
c_worktree=${C_WORKTREE:-/home/astesia/Sevenova/.v3-scoop-golden-C-seeded}
d_worktree=${D_WORKTREE:-/home/astesia/Sevenova/.v3-scoop-golden-D-seeded}
runtime_worktree=${RUNTIME_WORKTREE:-$d_worktree}
out=${OUTPUT_DIR:-/home/astesia/v3-scoop-seed-comparison/fixed-v3}
preload=${PRELOAD_LIBRARY:-/home/astesia/v3-scoop-seed-comparison/libv3_seed_random_numbers.so}
mkdir -p "$out" "$(dirname "$preload")"
g++ -std=c++17 -fPIC -shared "$root/tools/v3_seed_random_numbers.cpp" \
  -I/opt/ros/humble/include/random_numbers \
  -L/opt/ros/humble/lib -Wl,-rpath,/opt/ros/humble/lib -lrandom_numbers \
  -o "$preload"
mapfile -t seeds < <(/usr/bin/python3 -c \
  'import json,sys; print(*json.load(open(sys.argv[1]))["seeds"], sep="\n")' \
  "$root/tools/v3_scoop_golden_20260921/OMPL_SEEDS.json")
cd "$runtime_worktree" || exit 1
source tools/ros_humble_env.sh || exit 1
source ros2_ws/install/setup.bash || exit 1
export ROS_DOMAIN_ID=199 LD_PRELOAD="$preload"
: > "$out/matrix-status.tsv"
for seed in "${seeds[@]}"; do
  for variant in C D; do
    if [[ $variant == C ]]; then src=$c_worktree; else src=$d_worktree; fi
    dir="$out/$variant/seed-$seed"; mkdir -p "$dir"
    stem="$dir/v3-scoop-$variant-seed-$seed"
    /usr/bin/python3 "$src/ros2_ws/src/alfa_robot_moveit_config/scripts/v3_scoop_5x5_grasp_sequence_rerun.py" \
      --save "$stem.rrd" --ompl-seed "$seed" --no-spawn --no-resume \
      --planning-timeout 15 --rrt-retries 2 --limit-boxes 25 >"$dir/run.log" 2>&1
    printf '%s\t%s\t%s\n' "$seed" "$variant" "$?" >> "$out/matrix-status.tsv"
  done
done
