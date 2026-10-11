"""Explicit Ilia/EXPO coordinates through physical23D targets, with no clipping.

Ilia native normalization is distinct from the released BEHAVE critic frame.
These conversions preserve per-timestamp scales, native delta mask, and native
physical-gripper state transform. They do not alter the controller action.
"""

import json
from pathlib import Path
import numpy as np
from adapters.task0_rl import ActionCoordinates

ILIA_DELTA = np.array([3, 4, 5, *range(7, 14), *range(15, 22)])


class IliaExpoCoordinates:
    def __init__(
        self, ilia_checkpoint, critic_checkpoint, critic_asset="turning_on_radio"
    ):
        self.critic = ActionCoordinates.load(Path(critic_checkpoint), critic_asset)
        native = json.loads(
            (
                Path(ilia_checkpoint)
                / "assets/IliaLarchenko/behavior_224_rgb/norm_stats.json"
            ).read_text()
        )["norm_stats"]
        student = json.loads(
            (
                Path(critic_checkpoint) / "assets" / critic_asset / "norm_stats.json"
            ).read_text()
        )["norm_stats"]
        self.state_mean = np.asarray(native["state"]["mean"], np.float32)[:23]
        self.state_std = np.asarray(native["state"]["std"], np.float32)[:23]
        self.action_mean = np.asarray(
            native["actions"]["per_timestamp_mean"], np.float32
        )[:, :23]
        self.action_std = np.asarray(
            native["actions"]["per_timestamp_std"], np.float32
        )[:, :23]
        self.critic_state_mean = np.asarray(student["state"]["mean"], np.float32)
        self.critic_state_std = np.asarray(student["state"]["std"], np.float32)
        assert self.action_mean.shape == (30, 23) and self.action_std.shape == (30, 23)

    def native_state(self, physical_reduced):
        s = np.array(physical_reduced, copy=True, dtype=np.float32)
        s[..., 14] = 20 * s[..., 14] - 1
        s[..., 22] = 20 * s[..., 22] - 1
        return np.pad(
            (s - self.state_mean) / (self.state_std + 1e-6),
            [(0, 0)] * (s.ndim - 1) + [(0, 9)],
        )

    def critic_state(self, physical_reduced):
        s = (np.asarray(physical_reduced) - self.critic_state_mean) / (
            self.critic_state_std + 1e-6
        )
        return np.pad(s, [(0, 0)] * (s.ndim - 1) + [(0, 9)])

    def native_encode(self, physical, anchor):
        x = np.array(physical, copy=True, dtype=np.float32)
        h = x.shape[-2]
        assert h <= 30
        x[..., ILIA_DELTA] -= np.expand_dims(np.asarray(anchor)[..., ILIA_DELTA], -2)
        x = (x - self.action_mean[:h]) / (self.action_std[:h] + 1e-6)
        return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, 9)])

    def native_decode(self, encoded, anchor):
        x = np.asarray(encoded)[..., :23]
        h = x.shape[-2]
        x = x * (self.action_std[:h] + 1e-6) + self.action_mean[:h]
        x = x.copy()
        x[..., ILIA_DELTA] += np.expand_dims(np.asarray(anchor)[..., ILIA_DELTA], -2)
        return x

    def native_to_critic(self, encoded, anchor):
        return self.critic.encode(self.native_decode(encoded, anchor), anchor)

    def critic_to_native(self, encoded, anchor):
        return self.native_encode(self.critic.decode(encoded, anchor), anchor)
