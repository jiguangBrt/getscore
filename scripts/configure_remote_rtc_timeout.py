"""Expose the existing OpenPI client's shared timeout flags in its RTC smoke CLI."""

import shutil
from pathlib import Path

path = Path("/mnt/reacher-fast/openpi_ur_pp_202607/repo/scripts/marvinpro_rtc_smoke.py")
source = path.read_text()
if "--request-timeout-s" not in source:
    old = "    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)"
    if source.count(old) != 1:
        raise SystemExit("Unexpected RTC smoke source; no changes made")
    backup = path.with_name(path.name + ".bak-getscore-20261019")
    if backup.exists():
        raise SystemExit("Backup already exists; no changes made")
    shutil.copy2(path, backup)
    source = source.replace(
        "    args = parser.parse_args()",
        '    parser.add_argument("--connect-timeout-s", type=float, default=20.0)\n'
        '    parser.add_argument("--request-timeout-s", type=float, default=300.0)\n'
        "    args = parser.parse_args()",
    ).replace(
        old,
        "    policy = websocket_client_policy.WebsocketClientPolicy(\n"
        "        args.host, args.port,\n"
        "        connect_timeout_s=args.connect_timeout_s,\n"
        "        request_timeout_s=args.request_timeout_s,\n"
        "    )",
    )
    path.write_text(source)
    print("Configured shared timeout flags; backup:", backup)
else:
    print("RTC smoke CLI already exposes request timeout")
