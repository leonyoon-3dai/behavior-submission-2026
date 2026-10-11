"""Build upstream RealTimeEXPOFTLearner with a frozen native Ilia actor."""

from pathlib import Path
import jax
import numpy as np
from openpi.training import sharding
from openpi.shared import normalize
from b1k.training import config
from b1k.policies.policy_config import create_trained_policy
from adapters.ilia_expo_actor import FrozenIliaActor, FrozenActorState
from expo_ft.agents.alg.realtime_expo_ft import RealTimeEXPOFTLearner

ROOT = Path(__file__).resolve().parents[1]
CRITIC_CHECKPOINT = (
    ROOT / "checkpoints/behave/turning_on_radio/behave_pi05_turn_on_the_radio"
)
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def build(checkpoint, critic_checkpoint=None, critic_asset="turning_on_radio"):
    critic_checkpoint = (
        Path(critic_checkpoint) if critic_checkpoint is not None else CRITIC_CHECKPOINT
    )
    cfg = config.get_config("pi_behavior_b1k_fast")
    policy = create_trained_policy(cfg, checkpoint)
    mesh = sharding.make_mesh(1)
    ds = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    rs = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    norm = normalize.load(critic_checkpoint / "assets" / critic_asset)
    actor = FrozenIliaActor(
        policy, checkpoint, critic_checkpoint, mesh, ds, rs, norm, critic_asset
    )
    state = FrozenActorState(params=actor.model_state)
    agent = RealTimeEXPOFTLearner.create(
        0,
        np.zeros((224, 224, 9), np.float32),
        np.zeros(23, np.float32),
        np.zeros(23, np.float32),
        actor=actor,
        actor_train_state=state,
        target_actor_params={},
        action_horizon=30,
        mesh=mesh,
        data_sharding=ds,
        replicated_sharding=rs,
        train_base_actor=False,
        freeze_encoder=True,
        freeze_critic_encoder=False,
        critic_lr=3e-4,
        actor_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(128, 128),
        num_qs=2,
        num_min_qs=2,
        N=2,
        n_edit_samples=1,
        edit_scale=0.01,
        discount=0.999,
        tau=0.005,
        latent_dim_image=64,
        latent_dim_state=32,
        encoder_stage_sizes=(1, 1, 1, 1),
        encoder_num_filters=16,
        critic_camera_keys=CAMERAS,
        use_full_augmentation=False,
        delay=4,
        replan_steps=15,
        resize_size=224,
        p1_use_prefix_conditioning=False,
        filter_N=2,
        filter_num_qs=2,
        filter_n_edit=1,
        filter_add_delayed_obs=True,
        init_temperature=0.01,
        default_prompt="turn on the radio",
        resume=False,
    )
    return agent, actor, policy, ds


PARTS = (
    "critic",
    "target_critic",
    "filter_critic",
    "batch_encoder",
    "edit_actor",
    "temp",
)


def auxiliary_tree(agent):
    return {
        "rng": agent.rng,
        **{
            name: {
                "params": getattr(agent, name).params,
                "opt_state": getattr(agent, name).opt_state,
                "step": getattr(agent, name).step,
            }
            for name in PARTS
        },
    }


def restore_auxiliary(agent, tree):
    changes = {"rng": tree["rng"]}
    for name in PARTS:
        changes[name] = getattr(agent, name).replace(**tree[name])
    return agent.replace(**changes).cache_infer_params()


def load_auxiliary(agent, path):
    """Restore optimizer namedtuples against the live template, not metadata lists."""
    import orbax.checkpoint as ocp

    template = jax.device_get(auxiliary_tree(agent))
    tree = ocp.PyTreeCheckpointer().restore(
        str(Path(path).resolve()),
        item=template,
        restore_args=jax.tree.map(
            lambda _: ocp.RestoreArgs(restore_type=np.ndarray), template
        ),
    )
    return restore_auxiliary(agent, tree)
