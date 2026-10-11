"""Capture native v3.9.3 controller transitions without changing policy execution."""
import json
import time
from pathlib import Path

import numpy as np


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


class TeacherCapture:
    def __init__(self, root, cameras, proprio_key, capacity, metadata, store_images=True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.cameras, self.proprio_key = cameras, proprio_key
        self.capacity, self.count = capacity, 0
        self.store_images = store_images
        self.metadata = dict(metadata)
        self.arrays = {}
        specs = {"raw_state": ((capacity+1, 61), np.float32),
                 "actions": ((capacity, 23), np.float32),
                 "rewards": ((capacity,), np.float32),
                 "dones": ((capacity,), bool), "truncated": ((capacity,), bool),
                 "timestamp": ((capacity+1,), np.float64),
                 "wall_timestamp": ((capacity+1,), np.float64)}
        if store_images:
            specs.update({f"image_{i}": ((capacity+1, 224, 224, 3), np.uint8) for i in range(3)})
        for key, (shape, dtype) in specs.items():
            self.arrays[key] = np.lib.format.open_memmap(self.root/f"{key}.npy", mode="w+", dtype=dtype, shape=shape)
        self.events = (self.root/"events.jsonl").open("w")
        self.save(False, None)

    def observation(self, obs, index):
        raw = array(obs[self.proprio_key])
        if raw.shape != (61,) or not np.isfinite(raw).all():
            raise ValueError(f"Invalid proprio {raw.shape}")
        self.arrays["raw_state"][index] = raw
        for i, role in enumerate(("head", "left_wrist", "right_wrist")):
            image = array(obs[self.cameras[role]+"::rgb"])[..., :3]
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f"Unexpected native RGB {image.shape} {image.dtype}")
            if self.store_images:
                self.arrays[f"image_{i}"][index] = image
            elif index == 0:
                np.save(self.root/f"initial_image_{i}.npy", image)
        self.arrays["timestamp"][index] = index/30.
        self.arrays["wall_timestamp"][index] = time.time()

    def before_step(self, obs):
        if self.count >= self.capacity:
            raise ValueError("Capture capacity exceeded")
        # Store precisely the observation used by the policy before this action.
        self.observation(obs, self.count)

    def after_step(self, action, reward, terminated, truncated, obs, diagnostics):
        action = array(action)
        if action.shape != (23,) or not np.isfinite(action).all() or not np.isfinite(float(reward)):
            raise ValueError("Invalid actual controller action / reward")
        self.arrays["actions"][self.count] = action
        self.arrays["rewards"][self.count] = float(reward)
        self.arrays["dones"][self.count] = bool(terminated)
        self.arrays["truncated"][self.count] = bool(truncated)
        self.observation(obs, self.count+1)
        self.events.write(json.dumps({"step": self.count, "episode_start": self.count == 0,
            "episode_end": bool(terminated or truncated), "timestamp": self.count/30.,
            "wall_timestamp": time.time(), "reward": float(reward),
            "terminated": bool(terminated), "truncated": bool(truncated), **diagnostics})+"\n")
        self.count += 1
        if self.count % 64 == 0 or terminated or truncated:
            self.save(bool(terminated or truncated), diagnostics.get("official_success"))

    def save(self, complete, success):
        for value in self.arrays.values():
            value.flush()
        self.events.flush()
        data = {**self.metadata, "transitions": self.count, "observations": self.count+1,
            "complete": complete, "official_success": success, "capacity": self.capacity,
            "action_source": "actual 23D tensor passed to env.step after policy wrapper",
            "rgb_arrays_stored": self.store_images,
            "rgb_recording": "full arrays" if self.store_images else "initial RGB arrays plus native full video; evaluation-only, not BC replay",
            "camera_order": ["head", "left_wrist", "right_wrist"], "control_hz": 30,
            "episode_boundaries": [0, self.count], "unused_rows_are_data": False}
        tmp = self.root/"manifest.tmp"
        tmp.write_text(json.dumps(data, indent=2)+"\n")
        tmp.replace(self.root/"manifest.json")

    def close(self, success):
        self.save(True, success)
        self.events.close()
