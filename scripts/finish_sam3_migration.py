"""Download the authorized SAM3 checkpoint, deploy, and verify segmentation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
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
    args = parser.parse_args()
    record = args.record.resolve()
    with (record / '.sam3-migration.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(status='downloading_sam3', updated_at_unix=time.time())

        def save(**values):
            status.update(values, updated_at_unix=time.time())
            temporary = record / 'sam3-supervisor.tmp'
            temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
            temporary.replace(record / 'sam3-supervisor.json')

        save()
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
            revision = json.loads((record / 'sam3-source.json').read_text())['sha']
            metadata = get_hf_file_metadata(hf_hub_url('facebook/sam3', 'sam3.pt', revision=revision),
                                            token=True, timeout=20)
            expected = metadata.etag.strip('"')
            assert len(expected) == 64
            save(source='facebook/sam3', revision=revision, expected_sha256=expected,
                 expected_bytes=metadata.size)
            path = Path(hf_hub_download('facebook/sam3', 'sam3.pt', revision=revision,
                                        token=True, local_dir=ROOT / 'data/sam3-checkpoint'))
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected or path.stat().st_size != metadata.size:
                raise RuntimeError('SAM3 download differs from the official size or SHA-256')
            save(status='transferring_sam3', local_checkpoint=str(path), sha256=expected)
            existing = ssh(f"python3 -c 'import hashlib; from pathlib import Path; "
                f"p=Path(\"{REMOTE}/checkpoints/sam3.pt\"); h=hashlib.sha256(); "
                "f=p.open(\"rb\"); "
                "[h.update(b) for b in iter(lambda:f.read(8*1024*1024),b\"\")]; f.close(); "
                f"assert h.hexdigest()==\"{expected}\"'", timeout=300)
            if existing.returncode == 0:
                save(status='remote_sam3_checkpoint_verified')
            else:
                transferred = subprocess.run(['rsync', '-a', '--partial', str(path),
                    f'{HOST}:{REMOTE}/checkpoints/sam3.pt.part'],
                    stdin=subprocess.DEVNULL, capture_output=True, text=True)
                if transferred.returncode:
                    raise RuntimeError('SAM3 transfer failed: ' + transferred.stderr[-1500:])
            verified = existing if existing.returncode == 0 else ssh(f"python3 -c 'import hashlib; from pathlib import Path; "
                f"p=Path(\"{REMOTE}/checkpoints/sam3.pt.part\"); h=hashlib.sha256(); "
                "f=p.open(\"rb\"); "
                "[h.update(b) for b in iter(lambda:f.read(8*1024*1024),b\"\")]; f.close(); "
                f"assert h.hexdigest()==\"{expected}\"; "
                "p.replace(p.with_name(\"sam3.pt\")); print(\"SHA-256 verified\")'", timeout=300)
            if verified.returncode:
                raise RuntimeError('Remote SAM3 checksum verification failed: ' + verified.stderr[-1500:])
            while True:
                started = ssh(f'python3 {REMOTE}/services/start_migrated_inference.py sam3', timeout=60)
                if started.returncode == 0:
                    save(status='waiting_for_sam3_service', service_start=started.stdout)
                    break
                if 'No GPU meets' not in started.stderr:
                    raise RuntimeError('SAM3 startup failed: ' + started.stderr[-2000:])
                save(status='waiting_for_lightly_loaded_gpu')
                time.sleep(60)
            sys.path[:0] = [str(ROOT), str(ROOT / 'RPent')]
            from rpent.utils.rpc import wait_for_ready
            from rpent.utils.rpc.http_rpc import HttpRpcClient
            from rpent.robots.components.sam3_client import Sam3Client
            import imageio.v2 as imageio
            import numpy as np
            rpc = HttpRpcClient('http://127.0.0.1:18114')
            service_pid = json.loads(started.stdout.strip().splitlines()[-1])['pid']

            class RemoteService:
                name = 'remote SAM3'
                next_check = 0

                def poll(self):
                    if time.monotonic() < self.next_check:
                        return None
                    self.next_check = time.monotonic() + 10
                    checked = ssh(f"python3 -c 'from pathlib import Path; "
                        f"p=Path(\"/proc/{service_pid}/stat\"); "
                        "assert p.exists() and p.read_text().split()[2] != \"Z\"'", timeout=30)
                    return None if checked.returncode == 0 else checked.returncode

            wait_for_ready(rpc, timeout_s=900, poll_interval_s=5, daemon=RemoteService())
            save(status='verifying_real_scene_segmentation')
            image = record / 'gawm-smoke/initial.png'
            result = Sam3Client(rpc).segment(image.read_bytes(), text_prompt='basket')
            if not result.found or result.mask is None or not result.mask.any():
                raise RuntimeError(f'SAM3 did not segment the basket: {result.reason}')
            mask = result.mask
            actual_image = imageio.imread(image)
            assert mask.shape == actual_image.shape[:2]
            imageio.imwrite(record / 'sam3-basket-mask.png', np.asarray(mask, dtype=np.uint8) * 255)
            verification = dict(status='passed', image=str(image), prompt='basket', found=True,
                                 score=result.score, box=result.box, mask_shape=list(mask.shape),
                                 mask_pixels=int(mask.sum()), checkpoint_sha256=expected)
            (record / 'sam3-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
            save(status='sam3_complete', verification=verification)
        except Exception as exc:
            save(status='requires_infrastructure_review', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
