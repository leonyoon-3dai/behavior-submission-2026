# Evaluation README — robogoomy / Ilia + EXPO-RT

Author: Leonard Yoon. Affiliation: Independent. Final policy seed:0. Policy routes:000,001,002,003,005. Submitted self-evaluation panel:002,001,005 × public0–3 × one rollout, total12. Missing outputs count as zero. All final-panel results and videos are retained unchanged, including all failures.

## Public artifacts and image identity

- Docker image: `ghcr.io/leonyoon-3dai/behavior-submission-2026:submission12-20261011`
- Digest-pinned URI: `ghcr.io/leonyoon-3dai/behavior-submission-2026:submission12-20261011`
- Registry publication status: `pending`
- Anonymous registry pull verified: `False`
- Local image configuration ID: `sha256:3098b31a4a6b222bc4a1f23f879da6a72b54bc15d5cc89183b2fff99c413cf4e` (distinct from registry manifest digest).
- Final artifact repository: https://huggingface.co/leonayoon/behavior-submission-task001-002-005-public0-3-2026
- Result ZIP: https://huggingface.co/leonayoon/behavior-submission-task001-002-005-public0-3-2026/resolve/main/package/submission12-final.zip?download=true
- Original videos: [VIDEOS.md](VIDEOS.md), with public/native instance mapping and SHA256.
- No external model volume, HF token, proprietary API credential, or simulator asset is needed by the policy image. The simulator runs outside it and uses organizer-provided BEHAVIOR assets.

## Start the policy

Linux x86_64, NVIDIA driver and Docker NVIDIA Container Toolkit are required. Use one GPU per policy server, one evaluator connection, batch size1. Port8000 is a binary msgpack WebSocket endpoint; `/healthz` and `/metadata` are HTTP endpoints. Returned action has shape `(1,23)`; reset has no response. Action-chunk replay is disabled.

```bash
docker pull ghcr.io/leonyoon-3dai/behavior-submission-2026:submission12-20261011
docker run --rm --name robogoomy-policy --gpus device=0 --cpus 16 --shm-size 8g -p 127.0.0.1:8000:8000 ghcr.io/leonyoon-3dai/behavior-submission-2026:submission12-20261011
```

From a second terminal:

```bash
curl --fail http://127.0.0.1:8000/healthz
curl --fail http://127.0.0.1:8000/metadata
```

An optional writable log volume may be mounted at `/logs`. The default command loads `/models/model-manifest.json` and seed0. The server loads the relevant task route lazily on the first action; initial load/JIT is part of first-action latency. Healthcheck alone does not validate model inference. Do not add `--contract-smoke-only` for evaluation.

## Exact evaluator, wrapper, and robot

The original final panel used unmodified stock BEHAVIOR source revision `bd049de3119acdcdf2334fe9e1ebe060fa20c108` (v3.9.3-post1), R1Pro, and `omnigibson.eval.wrappers.RGBDFullResWrapper`. Wrapper head resolution720×720 and wrists480×480, RGB/depth/proprio provided. The policy consumes RGB/proprio, resizes RGB internally, and discards privileged fields and depth. No custom robot, scene manipulation, tiled rendering, action replay, or timeout override was used.

The ZIP includes the complete wrapper source (`wrapper.py` and `evaluation-wrapper-source/`), unchanged `r1pro.yaml`, and hash/source provenance. Install OmniGibson and its official licensed assets using the [official installation instructions](https://behavior.stanford.edu/getting_started/installation.html). From the pinned BEHAVIOR repository root, with its simulator environment activated and this ZIP extracted at `$B1K_SUBMISSION_DIR`, the full evaluator command for one instance is:

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

The other submitted task names are `picking_up_trash` (001) and `setting_mousetraps` (005). Run each selected public index0,1,2,3 once with one logical environment and the default task timeout. Public index0–3 maps to native IDs301–304; original JSON names/IDs are preserved. Use a fresh output directory and retain every result. These commands are reproducibility instructions, not a request to repeat the completed final panel.

## Method, data, and development exposure

The actor is the original released Ilia checkpoint2 for tasks000/001 and checkpoint1 for002/003/005. It is not the separately trained F1 actor. Actor weights stayed frozen during auxiliary adaptation. Each task uses20 official demonstrations, batch8, 1000 auxiliary updates. Inference samples2 actor candidates and1 residual edited candidate, scores with the minimum of target-critic heads, then selects the best of3. Edit scale0.01, one edit step, 15 executed actions, 4-action rolling carry, native stage voting and correction rules retained. Training-only FAST/filter-critic branches are not used for online candidate scoring. Details: [inference audit](docs/submission-2026/ILIA-RT-EXPO-INFERENCE-AUDIT.md).

Released pretrained weights and official demonstrations are reused; no additional human teleoperation, scripted-data collection, or new RL rollout collection was performed for this submission. Total upstream pretraining hours are not independently verified and are not claimed as newly collected data. Ilia weights: https://huggingface.co/IliaLarchenko/behavior_submission (Apache-2.0); implementation: https://github.com/IliaLarchenko/behavior-1k-solution. Third-party license notices are preserved in the image `/licenses` and upstream sources.

The six-task pilot previously used public index0 and training instances23/24. This development exposure is disclosed; no explicit additional organizer permission is claimed. The final frozen-policy panel was run once per submitted instance, no best-of selection. Pilot results are not inserted into the final ZIP.

## Runtime evidence and limits

Deployment check: `in_progress`. It uses synthetic RGB/proprio, tests finite35 actions per route, seeded reset, and route transitions with embedded weights. Physical rollouts added:0. Peak current-device memory: `pending` MiB. A real single24GB GPU test was skipped by user choice (`actual_24gb_tested=false`); current L40S measurements do not prove actual24GB support.

The final self-evaluation used four L40S GPUs with policy and simulator sharing each GPU, 46 logical CPUs, changing simulator cgroup quotas recorded during execution (7→10→10.5, 0/2→11→12, then remaining0/1/2→15). These are resource ceilings, not dedicated core counts. Policy quota4. Completed raw files record simulation steps/time; critic timing and periodic progress estimates are not exact official action-to-action timing traces. The official average1FPS/600-second round-trip requirement must be assessed using its stated measurement definition; no exact episode-level timing proof is claimed from periodic samples alone.

| Task | Public 0 | Public 1 | Public 2 | Public 3 | Mean final Q |
|---|---:|---:|---:|---:|---:|
| 002 | 0.5714 | 0.7143 | 0.5714 | 0.0000 | 0.4643 |
| 001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 005 | 0.0000 | 0.5000 | 0.0000 | 0.3333 | 0.2083 |

Full success0/12. Observed panel Q mean0.22420635; three-task/30-instance mean0.08968254; full100-task/1000-instance mean0.00269048. These self-reported quantities are not organizer-verified scores.

[Official evaluation rules](https://behavior.stanford.edu/challenge/evaluation.html) · [Submission guidelines](https://behavior.stanford.edu/challenge/submission.html).
