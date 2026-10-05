import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, ValueError):
        return None


def snapshot():
    cpus = {}
    system = {}
    for line in Path('/proc/stat').read_text().splitlines():
        fields = line.split()
        if fields[0].startswith('cpu'):
            cpus[fields[0]] = list(map(int, fields[1:9]))
        elif fields[0] in ('ctxt', 'procs_running', 'procs_blocked'):
            system[fields[0]] = int(fields[1])
    processes = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        raw = read_text(directory / 'stat')
        if raw is None:
            continue
        fields = raw[raw.rfind(')') + 2:].split()
        try:
            processes[directory.name] = dict(
                name=raw[raw.find('(') + 1:raw.rfind(')')],
                ticks=int(fields[11]) + int(fields[12]), start_ticks=int(fields[19]),
                rss_bytes=int(fields[21]) * os.sysconf('SC_PAGE_SIZE'), threads=int(fields[17]),
            )
        except (ValueError, IndexError):
            continue
    sensors = {}
    for directory in Path('/sys/class/hwmon').glob('hwmon*'):
        name = read_text(directory / 'name')
        for sensor in directory.glob('temp*_input'):
            value = read_text(sensor)
            if value is not None:
                label = read_text(directory / (sensor.name.replace('_input', '_label'))) or sensor.stem
                sensors[f'{name}/{label}'] = float(value) / 1000
    frequency = {}
    throttles = {}
    topology = {}
    for directory in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
        cpu = directory.name
        frequency[cpu] = read_text(directory / 'cpufreq/scaling_cur_freq')
        throttles[cpu] = {path.name: read_text(path) for path in (directory / 'thermal_throttle').glob('*count')}
        topology[cpu] = {field: read_text(directory / 'topology' / field)
                         for field in ('core_id', 'physical_package_id', 'thread_siblings_list')}
    return dict(at=time.monotonic(), cpus=cpus, processes=processes, sensors_c=sensors,
                frequency_khz=frequency, throttle_counts=throttles, topology=topology,
                system=system, pressure={name: read_text('/proc/pressure/' + name) for name in ('cpu', 'memory', 'io')},
                loadavg=read_text('/proc/loadavg'), meminfo=read_text('/proc/meminfo'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=float, default=20)
    parser.add_argument('--recovery', type=float, default=10)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('benchmark command required')
    started = time.monotonic()
    previous = snapshot()
    baseline = previous
    samples = []
    process = None
    benchmark_started = benchmark_ended = None
    log = args.output.with_suffix('.benchmark.log').open('w')
    while True:
        now = time.monotonic()
        if process is None and now - started >= args.baseline:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            benchmark_started = now - started
            print(json.dumps({'event': 'benchmark_started', 'pid': process.pid, 'time_s': benchmark_started}), flush=True)
        if process is not None and process.poll() is not None and benchmark_ended is None:
            benchmark_ended = now - started
            print(json.dumps({'event': 'benchmark_ended', 'returncode': process.returncode, 'time_s': benchmark_ended}), flush=True)
        if benchmark_ended is not None and now - started >= benchmark_ended + args.recovery:
            break
        time.sleep(1)
        current = snapshot()
        elapsed = current['at'] - previous['at']
        utilization = {}
        for cpu, counters in current['cpus'].items():
            if cpu not in previous['cpus']:
                continue
            delta = [value - old for value, old in zip(counters, previous['cpus'][cpu])]
            total = sum(delta)
            utilization[cpu] = dict(busy_percent=100 * (total - delta[3] - delta[4]) / max(1, total),
                                   iowait_percent=100 * delta[4] / max(1, total),
                                   irq_percent=100 * (delta[5] + delta[6]) / max(1, total))
        activity = []
        for pid, info in current['processes'].items():
            old = previous['processes'].get(pid)
            if old is None or old['start_ticks'] != info['start_ticks']:
                continue
            percent = 100 * (info['ticks'] - old['ticks']) / os.sysconf('SC_CLK_TCK') / elapsed
            if percent > 0 or (process is not None and pid == str(process.pid)):
                activity.append(dict(pid=int(pid), cpu_percent_one_core=percent, **info))
        activity.sort(key=lambda item: item['cpu_percent_one_core'], reverse=True)
        phase = 'baseline' if process is None else ('recovery' if benchmark_ended is not None else 'benchmark')
        samples.append(dict(time_s=current['at'] - started, phase=phase, elapsed_s=elapsed,
                            utilization=utilization, processes=activity,
                            **{key: current[key] for key in ('sensors_c', 'frequency_khz', 'throttle_counts', 'system', 'pressure', 'loadavg', 'meminfo')}))
        previous = current
        if len(samples) % 10 == 0:
            print(json.dumps({'time_s': samples[-1]['time_s'], 'phase': phase,
                              'cpu_busy_percent': utilization['cpu']['busy_percent'],
                              'top_processes': activity[:3]}), flush=True)
    log.close()
    args.output.write_text(json.dumps(dict(command=command, benchmark_pid=process.pid,
                                          benchmark_returncode=process.returncode,
                                          benchmark_started_s=benchmark_started,
                                          benchmark_ended_s=benchmark_ended,
                                          baseline=baseline, samples=samples), indent=2) + '\n')
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
