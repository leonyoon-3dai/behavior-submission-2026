# Evaluation README — robogoomy / Ilia + EXPO-RT

**Author:** Leonard Yoon · **Affiliation:** Independent · **Final policy seed:** 0

This guide reproduces the submitted policy and its **12 self-evaluation rollouts**: tasks 002, 001, and 005, each run once on public instances 0–3. The image supports routes 000, 001, 002, 003, and 005. Missing outputs count as zero; all final-panel results and videos, including failures, are retained unchanged.

## Public artifacts and image identity

- Docker image: `ghcr.io/leonyoon-3dai/behavior-submission-2026:submission12-20261011`
- Digest-pinned URI: `ghcr.io/leonyoon-3dai/behavior-submission-2026@sha256:5870a511c06fae3ee18289ed711131a9d2545aa9fd2c1e311b18980e4b07ad7e`
- Registry publication status: `registry_public_anonymous_digest_pull_verified`; anonymous pull verified: `True`.
- Local image configuration ID: `sha256:3098b31a4a6b222bc4a1f23f879da6a72b54bc15d5cc89183b2fff99c413cf4e` (distinct from registry manifest digest).
- Final artifact repository: https://huggingface.co/leonayoon/behavior-submission-task001-002-005-public0-3-2026
- Result ZIP: https://huggingface.co/leonayoon/behavior-submission-task001-002-005-public0-3-2026/resolve/main/package/submission12-final.zip?download=true
- Original videos: [VIDEOS.md](VIDEOS.md), with public/native instance mapping and SHA256.

Model weights are embedded. The policy image needs no external model volume, HF token, proprietary API credential, or simulator assets. The simulator runs separately with organizer-provided BEHAVIOR assets.

## Start the policy

Requires Linux x86_64, an NVIDIA driver, and Docker NVIDIA Container Toolkit.

| Setting | Requirement |
|---|---|
| Resources | One GPU per policy server |
| Evaluation | One evaluator connection; batch size 1 |
| Policy endpoint | Port 8000; binary msgpack over WebSocket |
| HTTP endpoints | `/healthz`, `/metadata` |
| Action / reset | Action shape `(1,23)`; reset sends no response |
| Action-chunk replay | Disabled |

```bash
docker pull ghcr.io/leonyoon-3dai/behavior-submission-2026@sha256:5870a511c06fae3ee18289ed711131a9d2545aa9fd2c1e311b18980e4b07ad7e
docker run --rm --name robogoomy-policy --gpus device=0 --cpus 16 --shm-size 8g -p 127.0.0.1:8000:8000 ghcr.io/leonyoon-3dai/behavior-submission-2026@sha256:5870a511c06fae3ee18289ed711131a9d2545aa9fd2c1e311b18980e4b07ad7e
```

From a second terminal:

```bash
curl --fail http://127.0.0.1:8000/healthz
curl --fail http://127.0.0.1:8000/metadata
```

The default command uses `/models/model-manifest.json` and seed 0. An optional writable log volume can be mounted at `/logs`.

The first action loads the task route and triggers initial load/JIT, adding to first-action latency. A passing healthcheck does **not** validate inference. Do not use `--contract-smoke-only` for evaluation.

## Exact evaluator, wrapper, and robot

| Component | Final-panel configuration |
|---|---|
| BEHAVIOR | Unmodified stock revision `bd049de3119acdcdf2334fe9e1ebe060fa20c108` (v3.9.3-post1) |
| Robot | R1Pro |
| Wrapper | `omnigibson.eval.wrappers.RGBDFullResWrapper` |
| Camera resolution | Head: 720×720; wrists: 480×480 |
| Wrapper outputs | RGB, depth, proprioception |
| Policy inputs | RGB (resized internally) and proprioception; depth and privileged fields discarded |

No custom robot, scene manipulation, tiled rendering, action replay, or timeout override was used.

The ZIP contains the complete wrapper source (`wrapper.py` and `evaluation-wrapper-source/`), unchanged `r1pro.yaml`, and hash/source provenance.

Install OmniGibson and its official licensed assets using the [official installation instructions](https://behavior.stanford.edu/getting_started/installation.html). Activate the simulator environment, extract the ZIP at `$B1K_SUBMISSION_DIR`, and run this command from the pinned BEHAVIOR repository root:

```bash
python -m omnigibson.eval.eval \
  --task-name putting_away_Halloween_decorations \
  --mode public_test \
  --instance-indices 0 \
  --num-envs 1 \
  --num-rollouts 1 \
  --host 127.0.0.1 \
  --port 8000 \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --robot-config "$B1K_SUBMISSION_DIR/r1pro.yaml" \
  --write-video \
  --headless \
  --output-dir outputs/robogoomy/task002-public0
```

| Submitted task | Evaluator task name |
|---|---|
| 002 | `putting_away_Halloween_decorations` |
| 001 | `picking_up_trash` |
| 005 | `setting_mousetraps` |

For each task, run public indices 0, 1, 2, and 3 once, with one logical environment and the default task timeout. Public indices 0–3 map to native IDs 301–304; original JSON names and IDs are preserved. Use a fresh output directory and retain every result.

These instructions support reproduction; the submitted final panel is already complete.

## Method, data, and development exposure

| Component | Configuration |
|---|---|
| Frozen actor | Original released Ilia checkpoint 2 for 000/001; checkpoint 1 for 002/003/005. The separately trained F1 actor is not used. |
| Auxiliary adaptation | Per task: 20 official demonstrations, batch size 8, 1000 updates; actor weights remain frozen |
| Candidate selection | 2 actor candidates + 1 residual-edited candidate; score each using the minimum of target-critic heads and select the best of 3 |
| Action editing | Scale 0.01; one edit step |
| Execution | 15 executed actions; 4-action rolling carry; native stage voting and correction rules retained |

Training-only FAST/filter-critic branches are excluded from online candidate scoring. See the [inference audit](docs/submission-2026/ILIA-RT-EXPO-INFERENCE-AUDIT.md).

This submission reuses released pretrained weights and official demonstrations. It adds no human teleoperation, scripted-data collection, or new RL rollout collection. Upstream pretraining hours have not been independently verified and are not claimed as newly collected data.

- Ilia weights: https://huggingface.co/IliaLarchenko/behavior_submission (Apache-2.0)
- Implementation: https://github.com/IliaLarchenko/behavior-1k-solution
- Third-party notices: preserved in the image at `/licenses` and in upstream sources.

**Development exposure:** the six-task pilot used public index 0 and training instances 23/24. No explicit additional organizer permission is claimed. The final frozen policy ran once per submitted instance, with no best-of selection. The final ZIP excludes pilot results.

## Runtime evidence and limits

**Synthetic deployment check: `PASS`.** With embedded weights and synthetic RGB/proprioception, it checks 35 finite actions per route, seeded reset, and route transitions. Physical rollouts added: 0. Peak current-device memory: `7317` MiB.

**Actual 24GB GPU compatibility remains untested.** The real single-24GB GPU test was skipped by user choice (`actual_24gb_tested=false`). L40S measurements do not establish 24GB compatibility.

The final self-evaluation used four L40S GPUs, with policy and simulator sharing each GPU, and 46 logical CPUs. Recorded simulator cgroup quota changes were 7→10→10.5, 0/2→11→12, then remaining 0/1/2→15; the policy quota was 4. These quotas are resource ceilings, not dedicated core counts.

Raw outputs record simulation steps and time. Critic timing and periodic progress estimates are not exact action-to-action timing traces. Compliance with the official average 1FPS / 600-second round-trip requirement must use the official measurement definition; periodic samples alone do not prove episode-level timing compliance.

## Self-evaluation results

| Task | Public 0 | Public 1 | Public 2 | Public 3 | Mean final Q |
|---|---:|---:|---:|---:|---:|
| 002 | 0.5714 | 0.7143 | 0.5714 | 0.0000 | 0.4643 |
| 001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 005 | 0.0000 | 0.5000 | 0.0000 | 0.3333 | 0.2083 |

**Full success: 0/12.** Missing outputs are counted as zero in the broader means below.

| Coverage | Mean final Q |
|---|---:|
| Observed 12-rollout panel | 0.22420635 |
| Three tasks / 30 instances | 0.08968254 |
| Full 100-task / 1000-instance benchmark | 0.00269048 |

These are self-reported results, not organizer-verified scores.

[Official evaluation rules](https://behavior.stanford.edu/challenge/evaluation.html) · [Submission guidelines](https://behavior.stanford.edu/challenge/submission.html).
