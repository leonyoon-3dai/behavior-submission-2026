"""Frozen2025 Ilia actor ABI for the actual upstream RT-EXPO learner.

The actor keeps its native30-step, stage logits, correlated noise and transforms.
EXPO critic/editor use a separate fixed23D delta frame, explicitly converted via
physical targets. Only the EXPO auxiliary networks are trainable in this test.
"""

from functools import partial
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx, struct
from adapters.ilia_expo_coordinates import IliaExpoCoordinates, ILIA_DELTA
from adapters.task0_rl import DELTA_INDICES
from b1k.models.observation import Observation


@struct.dataclass
class FrozenActorState:
    params: object
    ema_params: object = None
    step: object = 0


class FrozenIliaActor:
    def __init__(
        self,
        policy,
        ilia_checkpoint,
        critic_checkpoint,
        mesh,
        ds,
        rs,
        critic_norm,
        critic_asset="turning_on_radio",
    ):
        self.coords = IliaExpoCoordinates(
            ilia_checkpoint, critic_checkpoint, critic_asset
        )
        self.graph, self.model_state = nnx.split(policy._model)
        self.mesh = mesh
        self.data_sharding = ds
        self.replicated_sharding = rs
        self.infer_sharding = rs
        self.model_config = SimpleNamespace(action_dim=32, action_horizon=30)
        self.data_config = SimpleNamespace(norm_stats=critic_norm)
        self.freeze_pi05_encoder = True
        self.last_logits = None
        self.last_anchor = None
        self.cm = jnp.asarray(self.coords.critic.mean)
        self.cs = jnp.asarray(self.coords.critic.std) + 1e-6
        self.sm = jnp.asarray(self.coords.critic_state_mean)
        self.ss = jnp.asarray(self.coords.critic_state_std) + 1e-6
        self.nsm = jnp.asarray(self.coords.state_mean)
        self.nss = jnp.asarray(self.coords.state_std) + 1e-6
        self.nam = jnp.asarray(self.coords.action_mean)
        self.nas = jnp.asarray(self.coords.action_std) + 1e-6

    def get_params(self, state):
        return state.params

    def process_raw_inputs(self, observations, action_dim, resize_size):
        result = {k: v for k, v in observations.items() if v is not None}
        self.last_anchor = result["state"][..., :23] * self.ss + self.sm
        return result

    def process_transformed_outputs(self, actions):
        anchor = self.last_anchor
        out = actions[..., :23] * self.cs + self.cm
        return out.at[..., DELTA_INDICES].add(anchor[..., None, DELTA_INDICES])

    def _native(self, x):
        anchor = x["state"][..., :23] * self.ss + self.sm
        s = (
            anchor.at[..., 14]
            .set(20 * anchor[..., 14] - 1)
            .at[..., 22]
            .set(20 * anchor[..., 22] - 1)
        )
        s = (s - self.nsm) / self.nss
        s = jnp.pad(s, [(0, 0), (0, 9)])
        return {**x, "state": s}, anchor

    def _native_prefix(self, prefix, anchor):
        physical = prefix[..., :23] * self.cs + self.cm
        physical = physical.at[..., DELTA_INDICES].add(anchor[..., None, DELTA_INDICES])
        delta = physical.at[..., ILIA_DELTA].add(-anchor[..., None, ILIA_DELTA])
        d = prefix.shape[-2]
        out = (delta - self.nam[:d]) / self.nas[:d]
        return jnp.pad(out, [(0, 0), (0, 0), (0, 9)])

    @partial(jax.jit, static_argnames=("self", "num_samples"))
    def _sample(self, transformed, state, rng, num_samples, prefix=None, noise=None):
        x, anchor = self._native(transformed)
        b = x["state"].shape[0]
        x = jax.tree.map(lambda v: jnp.repeat(v, num_samples, axis=0), x)
        anchor = jnp.repeat(anchor, num_samples, axis=0)
        obs = Observation.from_dict(x)
        model = nnx.merge(self.graph, state.params)
        if prefix is not None:
            prefix = self._native_prefix(
                jnp.repeat(prefix, num_samples, axis=0), anchor
            )
        if noise is not None:
            white = noise.reshape(b * num_samples, 960)
            noise = (white @ model.action_correlation_cholesky.value.T).reshape(
                b * num_samples, 30, 32
            )
        actions, logits = model.sample_actions(
            rng, obs, noise=noise, initial_actions=prefix
        )
        physical = actions[..., :23] * self.nas + self.nam
        physical = physical.at[..., ILIA_DELTA].add(anchor[..., None, ILIA_DELTA])
        delta = physical.at[..., DELTA_INDICES].add(-anchor[..., None, DELTA_INDICES])
        return (delta - self.cm) / self.cs, logits

    def sample_actions(
        self, transformed_inputs, train_state, rng, train=False, num_samples=1
    ):
        actions, logits = self._sample(
            transformed_inputs, train_state, rng, num_samples
        )
        self.last_logits = np.asarray(logits)
        return actions, None

    def sample_actions_with_prefix(
        self, transformed_inputs, train_state, rng, prefix_padded, num_samples=1
    ):
        prefix = jnp.asarray(prefix_padded)[None]
        actions, logits = self._sample(
            transformed_inputs, train_state, rng, num_samples, prefix
        )
        self.last_logits = np.asarray(logits)
        return actions

    def sample_training_actions(
        self,
        transformed_inputs,
        train_state,
        rng,
        train=False,
        num_samples=1,
        noise=None,
    ):
        a, _ = self._sample(
            transformed_inputs, train_state, rng, num_samples, noise=noise
        )
        return a, None

    def sample_training_actions_with_prefix(
        self,
        transformed_inputs,
        train_state,
        rng,
        prefix_padded,
        num_samples=1,
        noise=None,
    ):
        a, _ = self._sample(
            transformed_inputs,
            train_state,
            rng,
            num_samples,
            jnp.asarray(prefix_padded),
            noise,
        )
        return a, None
