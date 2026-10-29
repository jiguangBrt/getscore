"""Run the original inference implementations with isolated paths and budgets."""
from pathlib import Path
import runpy
import sys

BASE = Path('/data/wangyihan/getscore-inference-20261028')
component = sys.argv[1]
if component == 'pi05':
    sys.path.insert(0, str(BASE / 'openpi/repo'))
    from scripts.serve_policy import Args, Checkpoint, EnvMode, create_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    args = Args(env=EnvMode.LIBERO, port=18008,
                policy=Checkpoint(config='pi05_libero', dir=str(BASE / 'checkpoints/pi05_libero')))
    policy = create_policy(args)
    WebsocketPolicyServer(policy=policy, host='127.0.0.1', port=18008,
                          metadata=policy.metadata).serve_forever()
elif component in ('gawm', 'sam3'):
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.05 if component == 'gawm' else 0.20)
    if component == 'gawm':
        sys.argv = ['gawm_policy_server.py', '--root', str(BASE / 'gawm'),
                    '--host', '127.0.0.1', '--port', '18115']
        runpy.run_path(str(BASE / 'services/gawm_policy_server.py'), run_name='__main__')
    else:
        sys.argv = ['sam3_server.py', '--transport', 'http',
                    '--host', '127.0.0.1', '--port', '18114']
        runpy.run_path(str(BASE / 'services/rpent/rpent/robots/components/sam3_server.py'),
                       run_name='__main__')
else:
    raise ValueError(component)
