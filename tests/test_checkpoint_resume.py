"""Verify checksum-protected checkpoint resume across HTTP range sizes."""
import base64
import hashlib
import io
import json
import re

import google_crc32c
import pytest

from scripts import download_openpi_checkpoint as downloader


@pytest.mark.parametrize('previous_chunk,done', [(2, [0, 2]), (8, [0])])
def test_resume_with_new_range_size_reuses_only_verified_bytes(tmp_path, monkeypatch, previous_chunk, done):
    data = b'abcdefghijk'
    crc = google_crc32c.Checksum(data)
    item = dict(name='checkpoints/pi05_libero/params/weights.bin', size=str(len(data)),
                generation='1', crc32c=base64.b64encode(crc.digest()).decode(),
                md5Hash=base64.b64encode(hashlib.md5(data).digest()).decode())
    target = tmp_path / 'params/weights.bin'
    target.parent.mkdir()
    partial = bytearray(b'X' * len(data))
    for offset in done:
        partial[offset:offset + previous_chunk] = data[offset:offset + previous_chunk]
    target.with_name(target.name + '.range.part').write_bytes(partial)
    target.with_name(target.name + '.ranges.json').write_text(json.dumps(
        dict(generation='1', chunk_bytes=previous_chunk, done=done)))
    requests = []

    def urlopen(request, **kwargs):
        if isinstance(request, str):
            return io.BytesIO(json.dumps(dict(items=[item])).encode())
        start, end = map(int, re.fullmatch(r'bytes=(\d+)-(\d+)', request.get_header('Range')).groups())
        requests.append(start)
        response = io.BytesIO(data[start:end + 1])
        response.status = 206
        response.headers = {'Content-Range': f'bytes {start}-{end}/{len(data)}'}
        response.geturl = lambda: request.full_url
        return response

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', urlopen)
    downloader.download(tmp_path, workers=1, chunk_bytes=4)
    assert target.read_bytes() == data
    assert 0 not in requests
    assert (tmp_path / 'download-manifest.json').is_file()
    assert not target.with_name(target.name + '.ranges.json').exists()
