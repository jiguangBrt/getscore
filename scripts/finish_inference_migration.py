"""Finish pi05 deployment after its remote checksum-verified download exits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/data/wangyihan/getscore-inference-20261028'
HOST = '49.233.66.228'


def ssh(command, timeout=None):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST, command],
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--download-pid', type=int, required=True)
    args = parser.parse_args()
    record = args.record.resolve()
    status = dict(status='waiting_for_pi05_download', download_pid=args.download_pid,
                  host=HOST, updated_at_unix=time.time())

    def save(**values):
        status.update(values, updated_at_unix=time.time())
        temporary = record / 'pi05-supervisor.tmp'
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(record / 'pi05-supervisor.json')

    save()
    try:
        # A kernel process-exit event, with no repeated model or SSH polling.
        command = ("python3 -c 'import os,select; "
                   f"fd=os.pidfd_open({args.download_pid}); select.select([fd],[],[]); os.close(fd)'")
        waited = ssh(command)
        save(download_wait_exit_code=waited.returncode,
             download_wait_stderr=waited.stderr[-1000:])
        check = ssh(f"python3 -c 'import json; from pathlib import Path; "
                    f"p=Path(\"{REMOTE}/checkpoints/pi05_libero\"); "
                    "m=json.loads((p/\"download-manifest.json\").read_text()); "
                    "assert len(m[\"objects\"])==16; "
                    "assert all((p/o[\"name\"].split(\"checkpoints/pi05_libero/\",1)[1]).stat().st_size==int(o[\"size\"]) for o in m[\"objects\"]); "
                    "print(\"16 checksum-verified objects present\")'", timeout=30)
        if check.returncode:
            raise RuntimeError('Checkpoint download did not finish: ' + check.stderr[-1500:])
        while True:
            started = ssh(f'python3 {REMOTE}/services/start_migrated_inference.py pi05', timeout=60)
            if started.returncode == 0:
                save(status='waiting_for_pi05_service', service_start=started.stdout)
                break
            if 'No GPU meets' not in started.stderr:
                raise RuntimeError('Pi05 startup failed: ' + started.stderr[-2000:])
            save(status='waiting_for_lightly_loaded_gpu')
            time.sleep(60)
        sys.path[:0] = [str(ROOT), str(ROOT / 'RPent')]
        from scripts.openpi_transport import OpenPiTransport
        end = time.monotonic() + 900
        last_error = None
        while time.monotonic() < end:
            transport = OpenPiTransport('127.0.0.1', 18008, connect_timeout_s=5, request_timeout_s=10)
            try:
                transport.metadata()
                break
            except Exception as exc:
                last_error = str(exc)
            finally:
                transport.close()
            time.sleep(5)
        else:
            raise RuntimeError('Pi05 did not become ready: ' + str(last_error))
        save(status='verifying_pi05_simulation_chain')
        env = {**os.environ}
        for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX'):
            env.pop(key, None)
        with (record / 'pi05-smoke.log').open('a') as log:
            checked = subprocess.run([str(ROOT / '.venv/bin/python'), '-u', '-m',
                'scripts.smoke_inference_228', '--policy', 'pi05', '--output', str(record / 'pi05-smoke')],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                timeout=900)
        if checked.returncode:
            raise RuntimeError('Pi05 simulation check failed; see pi05-smoke.log')
        verification = json.loads((record / 'pi05-smoke/verification.json').read_text())
        assert verification['status'] == 'passed' and verification['frames'] == verification['actions'] == 16
        save(status='pi05_complete', verification=verification)
    except Exception as exc:
        save(status='requires_infrastructure_review', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
