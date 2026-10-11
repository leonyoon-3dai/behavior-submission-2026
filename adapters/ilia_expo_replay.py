"""Critic-only replay with real full15-step windows and30-step actor metadata."""

import json
from pathlib import Path
import numpy as np
from adapters.task0_replay import Task0Replay
from adapters.task0_rl import reduce_state, IMAGE_NAMES as CAMERAS


class IliaCriticReplay:
    def __init__(self, root, teachers, negatives, actor, source, log):
        import pandas as pd

        self.task_id = 0
        self.stage_count = 5  # Native task000 count; historical Radio condition.
        self.replay = Task0Replay(root)
        self.actor = actor
        self.log = Path(log)
        self.rng = np.random.default_rng(0)
        self.lengths = (
            pd.read_parquet(source / "meta/episodes/chunk-000/file-000.parquet")
            .set_index("episode_index")["length"]
            .to_dict()
        )
        self.original = [
            i for i, (r, _) in enumerate(self.replay.episodes) if r["split"] == "train"
        ]
        self.positive = []
        self.negative = []
        self.draws = 0
        lookup = {
            r["task_instance_id"]: r
            for r, _ in self.replay.episodes
            if r["split"] == "train"
        }
        for pool, folders in [(self.positive, teachers), (self.negative, negatives)]:
            for folder in folders:
                folder = Path(folder)
                m = json.loads((folder / "manifest.json").read_text())
                assert (
                    m["complete"]
                    and m["instance_id"] in lookup
                    and m["instance_id"]
                    not in {23, 24, 25, 26, 282, 284, 285, 288, 296, 297, 299, 300, 301}
                )
                assert m["official_success"] == (pool is self.positive)
                n = m["transitions"]
                arrays = {
                    k: np.load(folder / f"{k}.npy", mmap_mode="r")
                    for k in [
                        "raw_state",
                        "actions",
                        "rewards",
                        "dones",
                        "truncated",
                        "image_0",
                        "image_1",
                        "image_2",
                    ]
                }
                assert all(
                    len(arrays[k]) >= n + 1
                    for k in ["raw_state", "image_0", "image_1", "image_2"]
                )
                r = {
                    **lookup[m["instance_id"]],
                    "transitions": n,
                    "source": "Ilia-success"
                    if pool is self.positive
                    else "Ilia-physical-failure",
                }
                pool.append(len(self.replay.episodes))
                self.replay.episodes.append((r, arrays))

    def locations(self, size):
        out = []
        for _ in range(size):
            x = self.rng.random()
            pool = (
                self.original
                if x < 0.7
                else self.positive
                if x < 0.85 or not self.negative
                else self.negative
            )
            e = int(self.rng.choice(pool))
            r, _ = self.replay.episodes[e]
            upper = r["transitions"] - 15
            assert upper >= 0
            x = self.rng.random()
            lower = (
                max(0, upper - 7) if x < 0.2 else max(0, upper - 200) if x < 0.5 else 0
            )
            out.append((e, int(self.rng.integers(lower, upper + 1))))
        return out

    def inputs(self, locations, offset=0):
        items = []
        c = self.actor.coords
        for e, start in locations:
            r, a = self.replay.episodes[e]
            n = r["transitions"]
            idx = min(start + offset, n)
            future = np.minimum(idx + np.arange(30), n - 1)
            anchor = reduce_state(a["raw_state"][idx])
            state = c.critic_state(anchor)
            actions = c.critic.encode(a["actions"][future], anchor)
            stage = min(
                self.stage_count - 1,
                int(idx / (self.lengths[r["episode_index"]] / self.stage_count)),
            )
            items.append(
                {
                    "state": state.astype(np.float32),
                    "actions": actions,
                    "image": {
                        k: np.asarray(a[f"image_{i}"][idx], np.float32) / 255 * 2 - 1
                        for i, k in enumerate(CAMERAS)
                    },
                    "image_mask": {k: np.True_ for k in CAMERAS},
                    "tokenized_prompt": np.array([self.task_id, stage], np.int32),
                    "tokenized_prompt_mask": np.array([True, True]),
                }
            )
        return __import__("jax").tree.map(lambda *x: np.stack(x), *items)

    def batch(self, size=8):
        locations = self.locations(size)
        x = self.inputs(locations)
        nxt = self.inputs(locations, 15)
        delayed = self.inputs(locations, 11)
        for key in ["image", "image_mask", "state"]:
            x["next_" + key] = nxt[key]
            x["next_delayed_" + key] = delayed[key]
        x["next_tokenized_prompt"] = nxt["tokenized_prompt"]
        x["next_delayed_tokenized_prompt"] = delayed["tokenized_prompt"]
        x["episode_step"] = np.asarray([t for _, t in locations], np.int32)
        x["actions"] = x["actions"].reshape(size, -1)
        rewards = []
        masks = []
        rows = []
        for e, t in locations:
            r, a = self.replay.episodes[e]
            stop = t + 15
            assert stop <= r["transitions"]
            rewards.append(
                sum(float(a["rewards"][i]) * 0.999 ** (i - t) for i in range(t, stop))
            )
            terminal = bool(
                np.any(a["dones"][t:stop])
                or (
                    r.get("source") == "Ilia-physical-failure"
                    and np.any(a["truncated"][t:stop])
                )
            )
            masks.append(float(not terminal))
            rows.append(
                {
                    "source": r.get("source", "original"),
                    "episode": r["episode_index"],
                    "instance": r["task_instance_id"],
                    "start": t,
                    "critic_valid_actions": 15,
                    "actor_metadata_padding": max(0, t + 30 - r["transitions"]),
                    "reward15": rewards[-1],
                    "bootstrap_mask": masks[-1],
                    "actor_update": False,
                }
            )
        x["rewards"] = np.asarray(rewards, np.float32)
        x["masks"] = np.asarray(masks, np.float32)
        x["valids"] = np.ones(size, np.float32)
        self.draws += size
        with self.log.open("a") as f:
            f.write(json.dumps(rows) + "\n")
        return x
