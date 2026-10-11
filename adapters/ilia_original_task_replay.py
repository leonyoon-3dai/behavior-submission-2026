"""Task-specific original replay with a frozen, preselected draw schedule.

Only needed RGB observations are stored as lossless PNGs, not padded arrays.
The critic still receives full physical15-action windows and real successors.
"""

import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from adapters.ilia_expo_replay import IliaCriticReplay


class PNGObservations:
    def __init__(self, folder, camera):
        self.folder, self.camera = folder, camera

    def __getitem__(self, index):
        path = self.folder / "rgb" / f"{int(index):06d}-{self.camera}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Missing preselected real observation: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class OriginalTaskCriticReplay(IliaCriticReplay):
    def __init__(self, root, actor, log):
        root = Path(root)
        m = json.loads((root / "manifest.json").read_text())
        assert m["complete"] and m["task_id"] in (0, 1, 2, 3, 4, 5, 7)
        self.task_id = m["task_id"]
        from b1k.models.pi_behavior_config import TASK_NUM_STAGES

        self.stage_count = TASK_NUM_STAGES[self.task_id]
        self.actor, self.log = actor, Path(log)
        self.draws = 0
        self.schedule_index = 0
        self.schedule = np.load(root / "draw-schedule.npy")
        self.lengths = {r["episode_index"]: r["source_rows"] for r in m["episodes"]}
        self.positive, self.negative = [], []
        episodes = []
        for r in m["episodes"]:
            assert r["task_instance_id"] not in m["excluded_instances"]
            folder = root / r["folder"]
            arrays = {
                k: np.load(folder / (k + ".npy"), mmap_mode="r")
                for k in ("raw_state", "actions", "rewards", "dones")
            }
            arrays.update({f"image_{i}": PNGObservations(folder, i) for i in range(3)})
            n = r["transitions"]
            assert len(arrays["raw_state"]) == n + 1
            assert all(len(arrays[k]) == n for k in ("actions", "rewards", "dones"))
            assert arrays["dones"][-1] and not arrays["dones"][:-1].any()
            episodes.append((r, arrays))
        self.replay = SimpleNamespace(episodes=episodes)
        self.original = list(range(len(episodes)))

    def locations(self, size):
        end = self.schedule_index + size
        assert end <= len(self.schedule), "Preregistered sampling draws exhausted"
        result = [
            tuple(map(int, row)) for row in self.schedule[self.schedule_index : end]
        ]
        self.schedule_index = end
        return result
