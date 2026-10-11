"""Serve an actually updated synchronous EXPO learner to the official evaluator."""
from collections import deque
import json
from pathlib import Path
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from openpi_client.image_tools import resize_with_pad

from adapters.task0_rl import ActionCoordinates, reduce_state


class BEHAVIORPolicy:
    def __init__(self, agent, checkpoint: Path, output: Path):
        self.agent = agent
        self.output = output
        self.coordinates = ActionCoordinates.load(checkpoint)
        self.queue = deque()
        self.anchor = None
        self.selection_mode = "edit"
        self.execution_steps = 8
        self.carry_prefix = False
        self.remaining_prefix = None
        # Each connection is serial. RT must use explicit per-inference anchors;
        # this synchronous wrapper is deliberately not reused for RT sampling.
        def denormalize(actor, actions, unnormalize=True):
            return self.coordinates.decode(np.asarray(actions), self.anchor)
        agent.actor.process_transformed_outputs = types.MethodType(denormalize, agent.actor)
        report_path = output / "report.json"
        if report_path.exists() and json.loads(report_path.read_text()).get("phase")=="multi_demo_offline_rl":
            self.validate_heldout()

    def validate_heldout(self):
        report=json.loads((self.output/"report.json").read_text())
        root=Path(report.get("replay_root",Path(__file__).resolve().parents[1]/"data/task0-replay-v2"))
        if not (root/'manifest.json').exists():
            (self.output/'serve-heldout-status.json').write_text(json.dumps({
                'status':'skipped_missing_replay',
                'note':'Checkpoint inference restored without training replay; use saved training validation, no fresh heldout loss claimed.'},indent=2)+'\n')
            return
        import jax
        from flax import nnx
        from adapters.task0_replay import Task0Replay
        from expo_ft.agents.alg.batch_utils import prepare_critic_batch
        replay=Task0Replay(root)
        @jax.jit
        def loss(state,batch):
            model=nnx.merge(state.model_def,state.params)
            model.eval()
            return model.compute_loss(jax.random.PRNGKey(17),batch[0],batch[1],train=False).mean()
        records=[]
        for index,(record,_) in enumerate(replay.episodes):
            if record["split"]!="heldout":continue
            locations=[(index,int((record["transitions"]-32)*fraction)) for fraction in (.25,.75)]
            data=replay.batch(self.agent.actor,locations,delay=0)
            prepared=prepare_critic_batch(data,32,23,23,32,8,self.agent.critic_camera_keys)
            value=float(loss(self.agent.actor_train_state,self.agent.actor.prepare_batch_for_actor(prepared)))
            if not np.isfinite(value):raise ValueError("Nonfinite heldout loss")
            records.append({"episode":record["episode_index"],"task_instance_id":record["task_instance_id"],
                "frame_indices":[i for _,i in locations],"actor_loss":value})
        (self.output/"heldout-validation.json").write_text(json.dumps({"records":records,
            "mean_actor_loss":float(np.mean([r["actor_loss"] for r in records])),
            "note":"Fixed heldout chunks, no gradient update; loss does not measure official task success."},indent=2)+"\n")

    def reset(self):
        self.queue.clear()
        self.remaining_prefix = None
        self.anchor = None

    def inputs(self, observation):
        raw = np.asarray(observation["robot_r1::proprio"])
        if raw.ndim != 1:
            raise ValueError("Initial experiment supports one simulator environment")
        inputs = {"observation/state": raw, "prompt": "turn on the radio"}
        for index, camera in enumerate(("zed_link", "left_realsense_link", "right_realsense_link")):
            key = f"robot_r1::robot_r1:{camera}:Camera:0::rgb"
            inputs[f"observation/image_{index}"] = resize_with_pad(np.asarray(observation[key])[..., :3], 224, 224)
        return inputs, reduce_state(raw)

    def act(self, observation):
        if not self.queue:
            inputs, self.anchor = self.inputs(observation)
            started = time.monotonic()
            sampler = self.agent.replace(n_edit_samples=0) if self.selection_mode=="select" else self.agent
            if self.carry_prefix and self.remaining_prefix is not None:
                import jax
                from expo_ft.agents.alg.batch_utils import extract_critic_fields
                transformed = self.agent.actor.process_raw_inputs(inputs, self.agent.action_dim, self.agent.resize_size)
                transformed = extract_critic_fields(transformed, self.agent.actor.model_config.action_dim,
                    self.agent.state_dim, self.agent.critic_camera_keys)
                key, rng = jax.random.split(self.agent.rng)
                cache = self.agent._infer_cache or {}
                train_state = cache.get("actor_train_state") or self.agent.actor_train_state
                prefix = self.coordinates.encode(self.remaining_prefix, self.anchor)
                normalized = self.agent.actor.sample_actions_with_prefix(transformed, train_state, key,
                    prefix_padded=prefix, num_samples=self.agent.N)
                actions = self.agent.actor.process_transformed_outputs(normalized)[0]
                self.agent = self.agent.replace(rng=rng)
                info = {"sample_time": 0.0, "pinned_prior_actions": len(prefix)}
            else:
                actions, self.agent, info = sampler.sample_actions(inputs, only_base_actions=self.selection_mode=="base")
            actions = np.asarray(actions).reshape(32, 23)
            if not np.isfinite(actions).all():
                raise ValueError("Nonfinite predicted actions")
            self.queue.extend(actions[:self.execution_steps])
            self.remaining_prefix = actions[self.execution_steps:self.execution_steps+4].copy() if self.carry_prefix else None
            record = {"phase": "online_inference", "seconds": time.monotonic()-started,
                "replan_steps": self.execution_steps, "carry_prefix": self.carry_prefix,
                "pinned_prior_actions": info.get("pinned_prior_actions", 0),
                "upstream_sample_time": float(info.get("sample_time", 0)),
                "timing_note": "seconds is measured wall time; upstream_sample_time may be a JIT-cached diagnostic"}
            with (self.output / "inference.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        return torch.from_numpy(np.array(self.queue.popleft(), dtype=np.float32))


class BEHAVIORRTPolicy(BEHAVIORPolicy):
    """Configurable overlap; report measured waits and prefix timing budgets."""
    def __init__(self, agent, checkpoint, output):
        super().__init__(agent, checkpoint, output)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="task0-rt-actor")
        self.future = None
        self.delay = int(agent.delay)

    def reset(self):
        if self.future is not None:
            self.future.result()
            self.future = None
        super().reset()

    def act(self, observation):
        if not self.queue:
            inputs, self.anchor = self.inputs(observation)
            started = time.monotonic()
            wait_seconds = 0.0
            slow_seconds = None
            delay = 0
            if self.future is None:
                sampler=self.agent.replace(n_edit_samples=0) if self.selection_mode=="select" else self.agent
                actions, self.agent, info = sampler.sample_actions(inputs,only_base_actions=self.selection_mode=="base")
            else:
                delay = self.delay
                wait_started = time.monotonic()
                precached, updated, old_anchor, slow_seconds = self.future.result()
                wait_seconds = time.monotonic() - wait_started
                self.future = None
                self.agent = updated
                precached = self.coordinates.rebase(np.asarray(precached), old_anchor, self.anchor)
                sampler=self.agent.replace(n_edit_samples=0) if self.selection_mode=="select" else self.agent
                actions, self.agent, info = sampler.sample_actions(inputs, precached=precached, delay=delay,
                    only_base_actions=self.selection_mode=="base")
            actions = np.asarray(actions).reshape(8, 23)
            if not np.isfinite(actions).all():
                raise ValueError("Nonfinite RT actions")
            self.queue.extend(actions)
            record = {"phase": "online_rt_inference", "seconds": time.monotonic()-started,
                "replan_steps": 8, "delay_control_steps": delay, "wait_seconds": wait_seconds,
                "slow_phase_seconds": slow_seconds,
                "slow_phase_exceeds_30hz_prefix_budget": slow_seconds > self.delay/30 if slow_seconds is not None else None,
                "slow_phase_not_ready": wait_seconds > .001,
                "note": "Configured overlap probe; not verified 30Hz real-time throughput."}
            with (self.output / "inference.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        if len(self.queue) == self.delay and self.future is None:
            delayed_inputs, old_anchor = self.inputs(observation)
            prefix = self.coordinates.encode(np.stack(tuple(self.queue)), old_anchor)
            snapshot = self.agent
            def infer():
                slow_started = time.monotonic()
                cached, updated = snapshot.sample_pre_cache(delayed_inputs, prefix_padded=prefix)
                cached = np.asarray(cached)  # Synchronize device work before measuring.
                return cached, updated, old_anchor, time.monotonic()-slow_started
            self.future = self.pool.submit(infer)
        return torch.from_numpy(np.array(self.queue.popleft(), dtype=np.float32))
