import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(report):
    windows = report["windows"]
    first = windows[:2]
    last = windows[-2:]

    def throughput(rows):
        return sum(row["states"] for row in rows) / sum(row["elapsed_s"] for row in rows)

    telemetry = report["telemetry"]

    def numbers(field):
        return [float(sample[field]) for sample in telemetry if sample[field] not in ("[N/A]", "N/A")]

    first_rate, last_rate = throughput(first), throughput(last)
    thermal_fields = ["clocks_throttle_reasons.sw_thermal_slowdown",
                      "clocks_throttle_reasons.hw_thermal_slowdown"]
    return dict(
        gpu=report["gpu"], duration_s=report["elapsed_s"], average_states_per_s=report["states_per_s"],
        first_interval_s=[first[0]["start_s"], first[-1]["end_s"]],
        last_interval_s=[last[0]["start_s"], last[-1]["end_s"]],
        first_states_per_s=first_rate, last_states_per_s=last_rate,
        change_percent=(last_rate / first_rate - 1) * 100,
        temperature_first_c=float(telemetry[0]["temperature.gpu"]),
        temperature_last_c=float(telemetry[-1]["temperature.gpu"]),
        temperature_max_c=max(numbers("temperature.gpu")),
        utilization_mean_percent=statistics.mean(numbers("utilization.gpu")),
        clock_first_mhz=float(telemetry[0]["clocks.sm"]), clock_last_mhz=float(telemetry[-1]["clocks.sm"]),
        power_mean_w=statistics.mean(numbers("power.draw")),
        thermal_active_samples=sum(any(sample[field] == "Active" for field in thermal_fields) for sample in telemetry),
        power_cap_active_samples=sum(sample["clocks_throttle_reasons.sw_power_cap"] == "Active" for sample in telemetry),
        thermal_report_values={field: sorted(set(sample[field] for sample in telemetry)) for field in thermal_fields},
        telemetry_samples=len(telemetry), telemetry_errors=report["telemetry_errors"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--ipc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = {"Local RTX 4060": json.loads(args.local.read_text()),
               "IPC RTX A2000": json.loads(args.ipc.read_text())}
    local, ipc = reports.values()
    for field in ("model_hash", "input_hash", "script_hash", "sphere_count", "scene_cuboids", "torch_version", "cuda_version"):
        if local[field] != ipc[field]:
            raise RuntimeError(f"Benchmark mismatch: {field}")
    for name, report in reports.items():
        if report["elapsed_s"] < 120 or not report["graph_reference_equal"] or not report["telemetry"]:
            raise RuntimeError(f"Incomplete stress run: {name}")
    summary = {name: summarize(report) for name, report in reports.items()}
    summary["ipc_fraction_of_local_throughput"] = ipc["states_per_s"] / local["states_per_s"]
    summary["matching_model_inputs_script_software"] = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for name, report in reports.items():
        axes[0].plot([(row["start_s"] + row["end_s"]) / 2 for row in report["windows"]],
                     [row["states_per_s"] / 1e6 for row in report["windows"]], "o-", label=name)
        for axis, field in zip(axes[1:], ("temperature.gpu", "clocks.sm", "power.draw")):
            axis.plot([sample["time_s"] for sample in report["telemetry"]],
                      [float(sample[field]) for sample in report["telemetry"]], label=name)
    for axis, label in zip(axes, ("Million states/s", "GPU temperature (C)", "SM clock (MHz)", "GPU power (W)")):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Continuous compute time (s)")
    figure.suptitle("cuRobo sustained FK + self/world collision, 15 DoF, 340 spheres\nSame 16384-state batches; CUDA Graph; 120-second target")
    figure.tight_layout()
    figure.savefig(args.output_dir / "comparison.png", dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
