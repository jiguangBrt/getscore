"""Start one isolated inference service on a lightly loaded GPU on 66.228."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

BASE = Path('/data/wangyihan/getscore-inference-20261028')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('component', choices=['pi05', 'gawm', 'sam3'])
    parser.add_argument('--sample-only', action='store_true')
    args = parser.parse_args()
    samples = []
    for index in range(3):
        output = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        rows = []
        for line in output.splitlines():
            gpu, uuid, total, used, util = [value.strip() for value in line.split(',')]
            rows.append(dict(index=int(gpu), uuid=uuid, total=int(total), used=int(used), util=int(util)))
        samples.append(rows)
        if index < 2:
            time.sleep(2)
    candidates = []
    required = {'pi05': 15000, 'sam3': 15000, 'gawm': 4000}[args.component]
    for index in range(len(samples[0])):
        gpu = samples[-1][index]
        utilization = sum(sample[index]['util'] for sample in samples) / len(samples)
        peak_used = max(sample[index]['used'] for sample in samples)
        if utilization <= 35 and gpu['total'] - peak_used >= required + 8192:
            candidates.append(dict(**gpu, mean_util=utilization, peak_used=peak_used))
    candidates.sort(key=lambda gpu: (gpu['peak_used'], gpu['mean_util']))
    record = dict(component=args.component, samples=samples, candidates=candidates,
                  recorded_at_unix=time.time(), memory_budget_mib=required, reserve_mib=8192)
    (BASE / f'records/{args.component}-gpu-selection.json').write_text(json.dumps(record, indent=2) + '\n')
    if args.sample_only:
        print(json.dumps(record, indent=2))
        return
    if not candidates:
        raise RuntimeError('No GPU meets the utilization and free-memory requirements; no service started')
    gpu = candidates[0]
    port = {'pi05': 18008, 'gawm': 18115, 'sam3': 18114}[args.component]
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    if args.component == 'pi05' and not (BASE / 'checkpoints/pi05_libero/download-manifest.json').is_file():
        raise RuntimeError('The official pi05 checkpoint is not yet verified')
    if args.component == 'sam3' and not (BASE / 'checkpoints/sam3.pt').is_file():
        raise RuntimeError('SAM3 checkpoint is missing')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu['uuid'], 'PYTHONNOUSERSITE': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'NO_ALBUMENTATIONS_UPDATE': '1',
           'OPENPI_DATA_HOME': str(BASE / 'checkpoints'), 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
           'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.20', 'JAX_COMPILATION_CACHE_DIR': str(BASE / 'cache/jax'),
           'SAM3_CHECKPOINT_PATH': str(BASE / 'checkpoints/sam3.pt')}
    for name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX'):
        env.pop(name, None)
    python = BASE / ('openpi/.venv/bin/python' if args.component == 'pi05' else 'vision310.venv/bin/python')
    command = [str(python), '-u', str(BASE / 'services/migrated_inference_entrypoint.py'), args.component]
    with (BASE / f'logs/{args.component}.log').open('a') as log:
        proc = subprocess.Popen(command, cwd=BASE, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                preexec_fn=lambda: os.nice(10))
    record.update(pid=proc.pid, gpu=gpu, command=command, port=port, status='starting')
    (BASE / f'records/{args.component}-service.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(component=args.component, pid=proc.pid, gpu=gpu['index'], port=port)))


if __name__ == '__main__':
    main()
