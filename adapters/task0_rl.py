"""BEHAVIOR R1 action coordinates shared by EXPO and Real-Time EXPO.

The anchor is the observation at the START of a predicted action chunk, not
each action's future observation. Explicit anchors also avoid sharing mutable
inference state between a real-time sampler and its training thread.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

CAMERAS = ("zed_link_camera_0", "left_realsense_link_camera_0", "right_realsense_link_camera_0")
IMAGE_NAMES = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
DELTA_INDICES = np.array([*range(3, 14), *range(15, 22)])


def rebase_normalized_jax(actions, old_state, new_state, norm_stats):
    """Rebase JAX action targets using normalized BEHAVIOR states (training)."""
    import jax.numpy as jnp
    state_stats = norm_stats["state"]
    action_stats = norm_stats["actions"]
    state_scale = jnp.asarray(state_stats.std) + 1e-6
    # The mean cancels when subtracting two observations in the same frame.
    difference = (old_state[..., :23] - new_state[..., :23]) * state_scale
    difference = difference / (jnp.asarray(action_stats.std) + 1e-6)
    offset = jnp.zeros_like(difference).at[..., DELTA_INDICES].set(difference[..., DELTA_INDICES])
    return actions.at[..., :23].add(jnp.expand_dims(offset, -2))


def reduce_state(raw):
    raw = np.asarray(raw)
    if raw.shape[-1] != 61:
        raise ValueError(f"Expected official 61D proprioception, got {raw.shape}")
    return np.concatenate((raw[..., :3], raw[..., 53:57], raw[..., 3:10],
                           raw[..., 24:26].sum(-1, keepdims=True), raw[..., 28:35],
                           raw[..., 49:51].sum(-1, keepdims=True)), axis=-1)


@dataclasses.dataclass(frozen=True)
class ActionCoordinates:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def load(cls, checkpoint: Path, asset_name="turning_on_radio"):
        stats = json.loads((checkpoint / "assets" / asset_name / "norm_stats.json").read_text())["norm_stats"]["actions"]
        return cls(np.asarray(stats["mean"], dtype=np.float32), np.asarray(stats["std"], dtype=np.float32))

    def encode(self, absolute, anchor):
        actions = np.array(absolute, dtype=np.float32, copy=True)[..., :23]
        anchor = np.asarray(anchor)[..., :23]
        actions[..., DELTA_INDICES] -= np.expand_dims(anchor[..., DELTA_INDICES], -2)
        actions = (actions - self.mean) / (self.std + 1e-6)
        return np.pad(actions, [(0, 0)] * (actions.ndim - 1) + [(0, 9)])

    def decode(self, normalized, anchor):
        actions = np.asarray(normalized)[..., :23] * (self.std + 1e-6) + self.mean
        actions = actions.copy()
        actions[..., DELTA_INDICES] += np.expand_dims(np.asarray(anchor)[..., DELTA_INDICES], -2)
        return actions

    def rebase(self, normalized, old_anchor, new_anchor):
        """Preserve absolute joint targets when an asynchronous observation changes."""
        result = np.array(normalized, copy=True)
        shift = (np.asarray(old_anchor)[..., :23] - np.asarray(new_anchor)[..., :23]) / (self.std + 1e-6)
        result[..., DELTA_INDICES] += np.expand_dims(shift[..., DELTA_INDICES], -2)
        return result


@dataclasses.dataclass(frozen=True)
class B1KInputs:
    def __call__(self, data):
        images = {}
        for index, name in enumerate(IMAGE_NAMES):
            image = np.asarray(data[f"observation/image_{index}"])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("Expected RGB HWC uint8 input")
            images[name] = image
        output = {"state": reduce_state(data["observation/state"]), "image": images,
                  "image_mask": {name: np.True_ for name in IMAGE_NAMES}}
        for key in ("prompt", "actions"):
            if key in data:
                output[key] = data[key]
        return output


@dataclasses.dataclass(frozen=True)
class DeltaActions:
    def __call__(self, data):
        if "actions" in data:
            data = dict(data)
            actions = np.array(data["actions"], copy=True)
            anchor = data["state"][..., DELTA_INDICES]
            if actions.ndim > anchor.ndim:
                anchor = np.expand_dims(anchor, -2)
            actions[..., DELTA_INDICES] -= anchor
            data["actions"] = actions
        return data


def register_openpi_config(checkpoint: Path):
    """Register into the active EXPO fork, retaining its modified π0.5 model."""
    from flax import nnx
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import normalize, nnx_utils
    from openpi.training import config

    @dataclasses.dataclass(frozen=True)
    class B1KData(config.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            return config.DataConfig(
                repo_id="behavior-1k/2026-challenge-demos", asset_id="turning_on_radio",
                norm_stats=normalize.load(checkpoint / "assets/turning_on_radio"),
                data_transforms=transforms.Group(inputs=[B1KInputs(), DeltaActions()]),
                model_transforms=config.ModelTransformFactory(default_prompt="turn on the radio")(model_config),
                use_quantile_norm=False,
            )

    name = "pi05_behave_task0_expo"
    config._CONFIGS_DICT[name] = config.TrainConfig(
        name=name, exp_name="task0", model=Pi0Config(pi05=True, action_horizon=32,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=B1KData(), freeze_filter=nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ema_decay=None, batch_size=2, wandb_enabled=False,
    )
    return name

