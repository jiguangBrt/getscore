"""Verify a migrated policy with real LIBERO-PRO observations and actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

# Reuse the recorder's pinned LIBERO-PRO renderer/asset environment.
from scripts.record_gawm_liberopro import write_json
import imageio.v2 as imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=['pi05', 'gawm'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=16)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from robots.libero.env_server import LiberoEnvFacade, make_env
    from scripts.openpi_transport import OpenPiTransport
    from scripts.openpi_rpent_bridge import OpenPiBridge
    from scripts.gawm_rpent_bridge import GawmBridge
    started = time.monotonic()
    port = 18008 if args.policy == 'pi05' else 18115
    transport = OpenPiTransport('127.0.0.1', port, connect_timeout_s=20,
                               request_timeout_s=300,
                               timing_path=output / 'network-timing.jsonl')
    facade = None
    try:
        metadata = transport.metadata()
        if args.policy == 'gawm':
            assert metadata['checkpoint_sha256'] == 'f698e76ae129ccfbe8349418acd79044f2a1ca6366d098546afb477fd79e7e13'
            assert metadata['strict_load'] == dict(missing_keys=[], unexpected_keys=[])
        bridge = (OpenPiBridge if args.policy == 'pi05' else GawmBridge)(transport)
        facade = LiberoEnvFacade(make_env(0, 0, 'libero_10_swap',
                                          max_episode_steps=args.max_steps), meta={})
        obs, _ = facade.reset()
        imageio.imwrite(output / 'initial.png', obs['main_images'])
        prompt = 'put both the alphabet soup and the tomato sauce in the basket'
        actions, raw_chunks, states, terminations, truncations = [], [], [], [], []
        video = output / 'episode.mp4'
        term = trunc = False
        with imageio.get_writer(video, fps=20, codec='libx264', quality=8,
                                macro_block_size=None) as writer:
            while len(actions) < args.max_steps and not (term or trunc):
                batch = {key: np.asarray(obs[key])[None]
                         for key in ('states', 'main_images', 'wrist_images')}
                batch['task_descriptions'] = [prompt]
                chunk = bridge.predict(batch)
                assert chunk.ndim == 3 and chunk.shape[0] == 1 and chunk.shape[-1] == 7
                assert np.isfinite(chunk).all()
                raw_chunks.append(chunk)
                for action in chunk[0, :args.max_steps - len(actions)]:
                    obs, reward, term, trunc, info = facade.step(action)
                    actions.append(action.copy())
                    states.append(np.asarray(obs['states']).copy())
                    terminations.append(bool(term));truncations.append(bool(trunc))
                    writer.append_data(np.ascontiguousarray(obs['main_images']))
                    if term or trunc:
                        break
        with imageio.get_reader(video) as reader:
            frames = sum(1 for _ in reader)
        assert frames == len(actions) > 0
        imageio.imwrite(output / 'final.png', obs['main_images'])
        np.savez_compressed(output / 'trajectory.npz', actions=np.asarray(actions),
                            states=np.asarray(states), raw_chunks=np.asarray(raw_chunks),
                            terminated=np.asarray(terminations), truncated=np.asarray(truncations))
        result = dict(status='passed', policy=args.policy, native_success=bool(term),
                      truncated=bool(trunc), actions=len(actions), frames=frames,
                      policy_queries=len(raw_chunks), video=str(video),
                      elapsed_s=round(time.monotonic() - started, 2),
                      scope='real policy/renderer/action/video chain; SAM3 and planner excluded',
                      metadata=metadata)
        write_json(output / 'verification.json', result)
        print(json.dumps({key: value for key, value in result.items() if key != 'metadata'}), flush=True)
    finally:
        if facade is not None:
            facade.close()
        transport.close()


if __name__ == '__main__':
    main()
