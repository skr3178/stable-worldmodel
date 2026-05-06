"""Small PushT expert collection -> data/pusht_expert_train.h5

Direct call to World.collect with format='hdf5'. Sized for a sanity-check
run (a few hundred episodes); scale up by editing EPISODES below.

Output: <repo_root>/data/pusht_expert_train.h5
"""

from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht import WeakPolicy


EPISODES = 200
NUM_ENVS = 8
IMAGE_SHAPE = (96, 96)
SEED = 0

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / 'data' / 'pusht_expert_train.h5'


def main():
    out = OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    world = swm.World(
        'swm/PushT-v1',
        num_envs=NUM_ENVS,
        image_shape=IMAGE_SHAPE,
        max_episode_steps=100,
        render_mode='rgb_array',
    )
    world.set_policy(WeakPolicy(dist_constraint=100, seed=SEED))

    print(f'Collecting {EPISODES} episodes -> {out}')
    world.collect(out, episodes=EPISODES, seed=SEED, format='hdf5')
    world.close()
    print(f'Done. Size: {out.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
