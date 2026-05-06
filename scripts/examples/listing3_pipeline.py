"""Runnable port of the paper's Listing 3 (Appendix A.1) end-to-end pipeline.

Differences from the paper listing (current API vs paper's API):
- World requires image_shape kwarg.
- world.record_dataset(dataset_name=...) -> world.collect(path, format='hdf5').
- your_expert_policy is the project's PushT WeakPolicy (random-near-block).
- world_model is a stub HeuristicCostModel implementing get_cost(info, actions).
  It scores action sequences by alignment with the agent->goal direction;
  enough to exercise CEM + WorldModelPolicy + evaluate without a trained WM.
"""

from pathlib import Path

import torch

import stable_worldmodel as swm
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
from stable_worldmodel.envs.pusht import WeakPolicy
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.solver import CEMSolver

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / 'data' / 'pusht_demo.h5'


class HeuristicCostModel(torch.nn.Module):
    """Stub world model: scores actions by alignment with goal direction.

    Real WMs roll out latent dynamics and compare to a goal embedding. Here we
    just need a Costable that returns (B, S) so the pipeline runs end-to-end.
    """

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        # action_candidates: (B, S, H, 2)
        pos_agent = info_dict['pos_agent'].float()      # (B, S, 1, 2)
        goal_pose = info_dict['goal_pose'].float()      # (B, S, 1, 3) xy+angle
        goal_xy = goal_pose[..., :2]                    # (B, S, 1, 2)

        goal_dir = goal_xy - pos_agent                  # (B, S, 1, 2)
        goal_dir = goal_dir / (goal_dir.norm(dim=-1, keepdim=True) + 1e-6)

        mean_action = action_candidates.mean(dim=2, keepdim=True)  # (B, S, 1, 2)
        mean_action = mean_action / (mean_action.norm(dim=-1, keepdim=True) + 1e-6)

        alignment = (mean_action * goal_dir).sum(dim=-1).squeeze(-1)  # (B, S)
        return -alignment


def main():
    image_shape = (64, 64)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists():
        DATASET_PATH.unlink()  # 'overwrite' mode

    # === 1. Collect a small dataset with the expert policy ===
    world = swm.World('swm/PushT-v1', num_envs=4, image_shape=image_shape)
    world.set_policy(WeakPolicy(dist_constraint=100, seed=0))

    print(f'\n[1/3] Recording 8 demo episodes -> {DATASET_PATH}')
    world.collect(DATASET_PATH, episodes=8, seed=0, format='hdf5')

    # === 2. Reload as HDF5Dataset (paper's Listing 3 step) ===
    print('\n[2/3] Reloading dataset')
    dataset = HDF5Dataset(
        path=DATASET_PATH,
        frameskip=1,
        num_steps=16,
        keys_to_load=['pixels', 'action', 'state'],
    )
    print(f'  episodes={len(dataset.lengths)}, total_steps={int(dataset.lengths.sum())}')
    sample = dataset[0]
    print(f'  sample keys: {list(sample.keys())}')
    for k, v in sample.items():
        print(f'    {k:8s} shape={tuple(v.shape)} dtype={v.dtype}')

    # === 3. Evaluate with CEM-based MPC ===
    print('\n[3/3] Evaluating WorldModelPolicy with CEMSolver')
    world_model = HeuristicCostModel()

    solver = CEMSolver(
        model=world_model,
        num_samples=64,
        n_steps=4,            # cheap: small CEM budget for the demo
        topk=8,
        device='cpu',
    )
    policy = WorldModelPolicy(
        solver=solver,
        config=PlanConfig(horizon=4, receding_horizon=2),
    )

    world.set_policy(policy)
    results = world.evaluate(episodes=4, seed=0)
    print(f"\nSuccess Rate: {results['success_rate']:.1f}%")
    print(f"  episode_successes: {results['episode_successes'].tolist()}")

    world.close()


if __name__ == '__main__':
    main()
