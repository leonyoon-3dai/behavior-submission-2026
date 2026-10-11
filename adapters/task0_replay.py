"""Episode-aware replay sampling with explicit terminal and chunk anchor semantics."""
import json
from pathlib import Path
import numpy as np


class Task0Replay:
    def __init__(self, root: Path, seed=0, online_roots=(), online_fraction=.25, timeout_terminal=False, actor_press_fraction=0):
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.rng = np.random.default_rng(seed)
        if not 0 <= actor_press_fraction <= 1:
            raise ValueError("Invalid actor press fraction")
        self.actor_press_fraction = actor_press_fraction
        self.focus_rng = np.random.default_rng(seed+1000)
        self.sampling_summary = {"actor_chunks":0,"terminal_action_chunks":0,"last_five_action_chunks":0,"focused_chunks":0,"episode_draws":{}}
        if not 0 <= online_fraction <= 1:raise ValueError("Invalid online replay fraction")
        self.online_fraction=online_fraction
        self.timeout_terminal=timeout_terminal
        self.episodes = []
        for record in self.manifest["episodes"]:
            arrays = {name: np.load(root / record["folder"] / (name+".npy"), mmap_mode="r")
                for name in ("raw_state", "state", "actions", "rewards", "dones", "image_0", "image_1", "image_2")}
            self.episodes.append((record, arrays))
        self.online_sources = []
        for folder in online_roots:
            folder = Path(folder)
            report = json.loads((folder / "manifest.json").read_text())
            if not report["complete"]:
                raise ValueError("Only completed online rollouts can enter replay")
            length = report["transitions"]
            if length < 1 or report["observations"] != length + 1:
                raise ValueError("Online replay has invalid observation count")
            arrays = {name: np.load(folder / (name+".npy"), mmap_mode="r")
                for name in ("raw_state", "actions", "rewards", "dones", "truncated", "image_0", "image_1", "image_2")}
            if any(len(arrays[key]) < length+1 for key in ("raw_state","image_0","image_1","image_2")):
                raise ValueError("Online replay is missing successor observations")
            record = {"split": "train", "transitions": length, "actor_eligible": False,
                "source": str(folder), "online": True}
            self.episodes.append((record, arrays))
            self.online_sources.append({"path":str(folder),"manifest":report})

    def locations(self, size, *, actor=False, split="train"):
        choices = [i for i,(r,_) in enumerate(self.episodes) if r["split"]==split and (not actor or r.get("actor_eligible",True))]
        result = []
        for _ in range(size):
            online_choices=[i for i in choices if self.episodes[i][0].get("online")]
            offline_choices=[i for i in choices if not self.episodes[i][0].get("online")]
            pool=online_choices if online_choices and not actor and self.rng.random()<self.online_fraction else offline_choices
            e = int(self.rng.choice(pool or choices))
            length = self.episodes[e][0]["transitions"]
            upper = length-32 if actor else length-1
            # Cover whole trajectories while emphasizing the final contact/press region.
            draw = self.rng.random()
            lower = max(0, upper-7) if not actor and draw<.2 else max(0, upper-200) if draw<.5 else 0
            start = int(self.rng.integers(lower, upper+1))
            if actor and self.actor_press_fraction and self.focus_rng.random() < self.actor_press_fraction:
                # Success-window proxy, not an inferred physical finger-contact label.
                # Half of added draws contain terminal action; half span nearby starts.
                start = upper if self.focus_rng.random() < .5 else int(self.focus_rng.integers(max(0,upper-31),upper+1))
                self.sampling_summary["focused_chunks"] += 1
            if actor:
                if upper < 0 or not 0 <= start <= upper:
                    raise ValueError("Actor sample is not a full valid action chunk")
                self.sampling_summary["actor_chunks"] += 1
                self.sampling_summary["terminal_action_chunks"] += int(start+31 == length-1)
                self.sampling_summary["last_five_action_chunks"] += int(start+31 >= length-5)
                key=str(self.episodes[e][0]["episode_index"])
                self.sampling_summary["episode_draws"][key]=self.sampling_summary["episode_draws"].get(key,0)+1
            result.append((e, start))
        return result

    def model_inputs(self, actor, locations, offset=0):
        items = []
        for episode, start in locations:
            record, arrays = self.episodes[episode]
            length = record["transitions"]
            index = min(start+offset, length)
            future = np.minimum(index+np.arange(32), length-1)
            raw = {"observation/state": arrays["raw_state"][index],
                "actions": np.asarray(arrays["actions"][future]), "prompt": "turn on the radio"}
            raw.update({f"observation/image_{i}": arrays[f"image_{i}"][index] for i in range(3)})
            transformed = actor.input_model_transforms(actor.input_transforms(raw))
            transformed["image"] = {k: v.astype(np.float32)/255*2-1 for k,v in transformed["image"].items()}
            items.append(transformed)
        return __import__('jax').tree.map(lambda *xs: np.stack(xs), *items)

    def batch(self, actor, locations, *, delay=0, discount=.99, replan=8):
        result = self.model_inputs(actor, locations)
        following = self.model_inputs(actor, locations, offset=replan)
        for key in ("image", "image_mask", "state"):
            result[f"next_{key}"] = following[key]
        if delay:
            delayed = self.model_inputs(actor, locations, offset=replan-delay)
            for key in ("image", "image_mask", "state"):
                result[f"next_delayed_{key}"] = delayed[key]
            result["next_delayed_actions"] = delayed["actions"].reshape(len(locations), -1)
        result["episode_step"] = np.asarray([i for _,i in locations], np.int32)
        result["actions"] = result["actions"].reshape(len(locations), -1)
        rewards, masks = [], []
        for episode, index in locations:
            record, arrays = self.episodes[episode]
            stop = min(index+replan, record["transitions"])
            rewards.append(sum(float(arrays["rewards"][j])*discount**(j-index) for j in range(index,stop)))
            terminal=np.any(arrays["dones"][index:stop])
            if self.timeout_terminal and record.get("online"):
                terminal=terminal or np.any(arrays["truncated"][index:stop])
            masks.append(float(not terminal))
        result["rewards"] = np.asarray(rewards,np.float32)
        result["masks"] = np.asarray(masks,np.float32)
        result["valids"] = np.ones(len(locations),np.float32)
        return result
