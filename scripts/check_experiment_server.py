"""Bounded SSH readiness check used by the server-recovery scheduled task."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'outputs/selected-original-2000step-20261026'
HOST = '192.168.50.73'


def check() -> dict:
    result = {
        'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'host': HOST,
        'tcp_22_reachable': False,
        'ssh_ready': False,
        'remote_changes': False,
    }
    try:
        with socket.create_connection((HOST, 22), timeout=5):
            result['tcp_22_reachable'] = True
    except OSError as exc:
        result['state'] = 'waiting_for_network'
        result['error'] = f'{type(exc).__name__}: {exc}'
        return result
    try:
        response = subprocess.run(
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6',
             '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=3',
             '-o', 'ServerAliveCountMax=1', f'user@{HOST}',
             'printf GETSCORE_SSH_READY'],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
        )
        result['ssh_ready'] = response.returncode == 0 and response.stdout == 'GETSCORE_SSH_READY'
        result['ssh_exit_code'] = response.returncode
        result['state'] = 'ready_to_resume_chat' if result['ssh_ready'] else 'waiting_for_ssh'
        if not result['ssh_ready']:
            result['error'] = response.stderr.strip()[-1200:]
    except subprocess.TimeoutExpired:
        result['state'] = 'waiting_for_ssh'
        result['error'] = 'SSH authentication or command exceeded 15 seconds'
    return result


def main() -> None:
    result = check()
    BATCH.mkdir(parents=True, exist_ok=True)
    destination = BATCH / 'server-monitor-last.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(destination)
    print(json.dumps(result, ensure_ascii=False))
    # Offline is an expected waiting state, not a failed scheduled run.


if __name__ == '__main__':
    main()
