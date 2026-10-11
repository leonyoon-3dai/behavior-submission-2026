"""Matched15-step Ilia BASE and learned RT-EXPO selection/edit inference."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from adapters.task0_rl import reduce_state
from adapters.ilia_expo_learner import CAMERAS
from expo_ft.agents.alg.realtime_expo_ft import _jitted_fast_select


class IliaExpoProxy:
    def __init__(self, agent, actor, policy, condition, log, inference_seed=0):
        self.inference_seed = inference_seed
        self.agent = agent.replace(
            rng=jax.random.PRNGKey(self.inference_seed),
            N=1 if condition == "BASE" else 2,
            n_edit_samples=0 if condition == "BASE" else 1,
        )
        self.actor = actor
        self.policy = policy
        self.condition = condition
        self.log = Path(log)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.calls = 0

    def __getattr__(self, key):
        return getattr(self.policy, key)

    def inputs(self, obs):
        raw = np.asarray(obs["observation/state"])
        anchor = reduce_state(raw)
        state = self.actor.coords.critic_state(anchor).astype(np.float32)
        names = ["egocentric_camera", "wrist_image_left", "wrist_image_right"]
        x = {
            "state": state[None],
            "image": {
                k: (
                    np.array(obs["observation/" + names[i]], copy=True).astype(
                        np.float32
                    )
                    / 255
                    * 2
                    - 1
                )[None]
                for i, k in enumerate(CAMERAS)
            },
            "image_mask": {k: np.array([True]) for k in CAMERAS},
            "tokenized_prompt": np.array(obs["tokenized_prompt"], copy=True)[None],
            "tokenized_prompt_mask": np.array(obs["tokenized_prompt_mask"], copy=True)[
                None
            ],
        }
        return x, anchor

    def slow(self, obs, prefix=None):
        x, anchor = self.inputs(obs)
        padded = (
            self.actor.coords.critic.encode(prefix, anchor)
            if prefix is not None
            else None
        )
        tick = time.monotonic()
        actions, agent = self.agent.sample_pre_cache(x, prefix_padded=padded)
        return (
            np.asarray(actions),
            agent,
            anchor,
            np.array(self.actor.last_logits, copy=True),
            time.monotonic() - tick,
        )

    def arm(self, wrapper, obs):
        if self.future is not None or wrapper.last_actions is None:
            return
        if len(wrapper.last_actions) - wrapper.action_index != 4:
            return
        prefix = np.array(wrapper.last_actions[wrapper.action_index :], copy=True)
        x = wrapper.prepare_batch_for_pi_behavior(wrapper.process_obs(obs))
        self.future = self.pool.submit(self.slow, x, prefix)

    def infer(self, obs, initial_actions=None, **kwargs):
        tick = time.monotonic()
        delay = 0
        if self.future is None:
            cached, agent, old_anchor, logits, slow = self.slow(obs, initial_actions)
        else:
            cached, agent, old_anchor, logits, slow = self.future.result()
            self.future = None
            delay = 4
        wait = time.monotonic() - tick
        x, anchor = self.inputs(obs)
        cached = self.actor.coords.critic.rebase(cached, old_anchor, anchor)
        self.agent = agent
        fast = time.monotonic()
        r = _jitted_fast_select(
            self.agent.rng,
            jnp.asarray(cached),
            x["image"],
            jnp.asarray(x["state"]),
            self.agent.batch_encoder.params,
            self.agent.edit_actor.params,
            self.agent.target_critic.params,
            encoder_fn=self.agent.batch_encoder.apply_fn,
            edit_fn=self.agent.edit_actor.apply_fn,
            critic_fn=self.agent.target_critic.apply_fn,
            N=self.agent.N,
            n_edit_samples=self.agent.n_edit_samples,
            num_qs=2,
            num_min_qs=2,
            replan_steps=15,
            action_dim=23,
            state_dim=23,
            padded_action_dim=32,
            delay=delay,
            edit_scale=0.01,
            edit_action_xyzg=False,
            freeze_encoder=True,
            only_base_actions=self.condition == "BASE",
            critic_camera_keys=CAMERAS,
        )
        selected = np.asarray(r["selected"])
        idx = 0 if r["idx"] is None else int(r["idx"])
        base = idx if idx < self.agent.N else idx - self.agent.N
        carry = cached[base, delay + 15 : delay + 19]
        actions = self.actor.coords.critic.decode(
            np.concatenate([selected, carry]), anchor
        )
        assert actions.shape == (19, 23) and np.isfinite(actions).all()
        self.agent = self.agent.replace(rng=r["rng"])
        self.calls += 1
        fast_s = time.monotonic() - fast
        row = {
            "call": self.calls,
            "condition": self.condition,
            "delay": delay,
            "slow_seconds": slow,
            "boundary_wait_seconds": wait,
            "fast_seconds": fast_s,
            "candidate_count": self.agent.N,
            "edit_samples": self.agent.n_edit_samples,
            "selected_idx": idx,
            "selected_edit": idx >= self.agent.N,
            "learned_Q_values": None
            if r["qs"] is None
            else np.asarray(r["qs"]).tolist(),
            "30Hz_prefix_budget_met": slow <= delay / 30 if delay else None,
            "stage_candidate": base,
            "physical_steps": 15,
            "carry": 4,
        }
        with self.log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return {
            "actions": actions,
            "subtask_logits": logits[base],
            "predicted_stage": int(np.argmax(logits[base])),
        }

    def reset(self):
        if self.future is not None:
            self.future.result()
            self.future = None
        self.calls = 0
        self.agent = self.agent.replace(rng=jax.random.PRNGKey(self.inference_seed))
        self.policy.reset()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
