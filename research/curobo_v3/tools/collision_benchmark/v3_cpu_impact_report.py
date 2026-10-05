import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    cpu = json.loads(args.cpu.read_text())
    gpu = json.loads(args.gpu.read_text())
    samples = cpu['samples']
    summary = {}
    previous_counts = cpu['baseline']['throttle_counts']
    for phase in ('baseline', 'benchmark', 'recovery'):
        selected = [sample for sample in samples if sample['phase'] == phase]
        totals = [sample['utilization']['cpu']['busy_percent'] for sample in selected]
        per_core = {name: [sample['utilization'][name]['busy_percent'] for sample in selected]
                    for name in selected[0]['utilization'] if name != 'cpu'}
        processes = {}
        for sample in selected:
            for process in sample['processes']:
                key = str(process['pid'])
                if key not in processes:
                    processes[key] = dict(name=process['name'], values=[])
                processes[key]['values'].append(process['cpu_percent_one_core'])
        process_summary = [dict(pid=int(pid), name=info['name'],
                                mean_cpu_one_core=sum(info['values']) / len(selected),
                                peak_cpu_one_core=max(info['values'])) for pid, info in processes.items()]
        process_summary.sort(key=lambda row: row['mean_cpu_one_core'], reverse=True)
        temperatures = [sample['sensors_c']['coretemp/Package id 0'] for sample in selected]
        summary[phase] = dict(samples=len(selected), cpu_mean_percent=statistics.mean(totals),
                              cpu_p95_percent=float(np.percentile(totals, 95)), cpu_peak_percent=max(totals),
                              per_cpu={name: dict(mean_percent=statistics.mean(values), peak_percent=max(values))
                                       for name, values in per_core.items()},
                              cpu_temperature_mean_c=statistics.mean(temperatures),
                              cpu_temperature_max_c=max(temperatures), processes=process_summary)
        current_counts = selected[-1]['throttle_counts']
        summary[phase]['package_throttle_delta'] = int(current_counts['cpu0']['package_throttle_count']) - int(previous_counts['cpu0']['package_throttle_count'])
        summary[phase]['per_cpu_frequency_mean_mhz'] = {
            name: statistics.mean(float(sample['frequency_khz'][name]) / 1000 for sample in selected)
            for name in names_from_topology(cpu)
        }
        previous_counts = current_counts
    last = samples[-1]['throttle_counts']
    first = cpu['baseline']['throttle_counts']
    changes = {name: {key: int(value) - int(first[name][key]) for key, value in counts.items()
                      if value is not None and first[name].get(key) is not None} for name, counts in last.items()}
    summary['cpu_throttle_counter_changes'] = changes
    summary['cpu_topology'] = cpu['baseline']['topology']
    summary['benchmark_pid'] = cpu['benchmark_pid']
    summary['benchmark_returncode'] = cpu['benchmark_returncode']
    summary['gpu_average_states_per_s'] = gpu['states_per_s']
    summary['gpu_duration_s'] = gpu['elapsed_s']
    assert cpu['benchmark_returncode'] == 0 and gpu['elapsed_s'] >= 120
    assert all(len(sample['utilization']) == 29 for sample in samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'cpu_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    timeline = [sample['time_s'] for sample in samples]
    names = sorted(summary['baseline']['per_cpu'], key=lambda name: int(name[3:]))
    figure, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    matrix = np.array([[sample['utilization'][name]['busy_percent'] for sample in samples] for name in names])
    heatmap = axes[0].imshow(matrix, aspect='auto', origin='lower', vmin=0, vmax=100,
                             extent=[timeline[0], timeline[-1], -0.5, 27.5], cmap='inferno')
    axes[0].set_ylabel('Logical CPU (0-27)')
    figure.colorbar(heatmap, ax=axes[0], label='CPU busy (%)', fraction=0.025, pad=0.01)
    axes[1].plot(timeline, [sample['utilization']['cpu']['busy_percent'] for sample in samples], label='Total CPU, 28 CPUs = 100%')
    axes[1].legend()
    axes[1].set_ylabel('Total CPU (%)')
    for pid, label in ((cpu['benchmark_pid'], 'cuRobo stress'), (2041, 'gnome-shell')):
        axes[2].plot(timeline, [next((process['cpu_percent_one_core'] for process in sample['processes'] if process['pid'] == pid), 0)
                               for sample in samples], label=label)
    axes[2].legend()
    axes[2].set_ylabel('Process CPU\n(1 core = 100%)')
    axes[3].plot(timeline, [sample['sensors_c']['coretemp/Package id 0'] for sample in samples], label='CPU package')
    offset = cpu['benchmark_started_s'] + gpu['warmup_capture_s']
    axes[3].plot([offset + sample['time_s'] for sample in gpu['telemetry']],
                 [float(sample['temperature.gpu']) for sample in gpu['telemetry']], label='GPU (aligned approximately)')
    axes[3].legend()
    axes[3].set_ylabel('Temperature (C)')
    for axis in axes:
        axis.axvline(cpu['benchmark_started_s'], color='green', linestyle='--')
        axis.axvline(cpu['benchmark_ended_s'], color='red', linestyle='--')
    axes[-1].set_xlabel('Monitoring time (s); green: process start, red: exit')
    figure.suptitle('IPC i7-14700: CPU impact during 120s cuRobo GPU stress\n20s background baseline + workload initialization/compute + 10s recovery')
    figure.tight_layout()
    figure.savefig(args.output_dir / 'cpu_impact.png', dpi=150)
    print(json.dumps({key: {field: value for field, value in row.items() if field not in ('per_cpu', 'processes')}
                      for key, row in summary.items() if key in ('baseline', 'benchmark', 'recovery')}, indent=2))


def names_from_topology(cpu):
    return sorted(cpu['baseline']['topology'], key=lambda name: int(name[3:]))


if __name__ == '__main__':
    main()
