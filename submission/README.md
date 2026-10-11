# Submission policy and runtime

Policy server for the **12 completed self-evaluation rollouts**: tasks **001, 002, and 005**, public instances **0–3**, once each.

**Start here:** [Evaluation and Docker launch instructions](../EVALUATION-README.md) · [Results overview](../README.md) · [Original videos](../VIDEOS.md)

## Task coverage

| Task | Activity | Frozen Ilia actor | Included in the 12-rollout panel |
|---|---|---|---|
| 000 | Turn on a radio | Checkpoint 2 | No |
| 001 | Pick up trash | Checkpoint 2 | Yes |
| 002 | Put away Halloween decorations | Checkpoint 1 | Yes |
| 003 | Clean up plates and food | Checkpoint 1 | No |
| 005 | Set mousetraps | Checkpoint 1 | Yes |

All five routes use **Ilia + EXPO-RT**, seed `0`, with matching actor, auxiliary weights, and normalization assets. Task 007 is pilot-only and excluded. **Unsupported tasks return zero actions**; no performance is claimed.

## Input and output

Use **one connection and one environment per server**, with no evaluator-side action-chunk replay.

| Field | Required shape or behavior |
|---|---|
| Robot state | Finite proprioception, `(1, 61)` |
| Camera input | Head and two wrist images; each `uint8`, `(1, H, W, 3)` or `(1, H, W, 4)` |
| Task ID | Integer array, `(1,)` or `(1, 1)` |
| Action output | Finite `float32`, `(1, 23)` |
| Reset | Clears episode state; no response message |

RGB is resized internally. Depth and privileged simulator fields are discarded.

## Code map

| File | Role |
|---|---|
| [`contract.py`](contract.py) | Validates inputs/outputs and switches task routes. |
| [`policy.py`](policy.py) | Loads Ilia and EXPO-RT; retains stage voting and correction rules. |
| [`server.py`](server.py) | Msgpack WebSocket on port `8000`; HTTP `/healthz` and `/metadata`. |
| [`auxiliary.py`](auxiliary.py) | Loads inference parameters without optimizer state. |
| [`scope.py`](scope.py) | Defines pilot, policy, and evaluation coverage. |
| [`model-manifest.final.json`](model-manifest.final.json) | Frozen routes and evaluation scope. |
| [`Dockerfile`](Dockerfile) | Runtime template; requires a prepared build context. |

EXPO-RT executes 15 uncompressed actions and carries four into the next prediction. See the [inference audit](../docs/submission-2026/ILIA-RT-EXPO-INFERENCE-AUDIT.md) for scheduling, candidate selection, and baseline differences.

## Release status

- **Public image:** `ghcr.io/leonyoon-3dai/behavior-submission-2026:submission12-20261011`.
- **Anonymous digest-pinned pull verified.** Launch instructions are linked above.
- **Weights embedded at `/models`.** No external model mount or Hugging Face token required. Optional logs: `/logs`.
- **Separate simulator**, using organizer-provided BEHAVIOR assets.
- **Lazy loading:** the first observation loads its route. `/healthz` alone does not validate inference.
- **Deployment checks passed; a physical 24 GB GPU remains untested.** L40S measurements do not establish 24 GB compatibility.

Use the published image for evaluation. Never enable `--contract-smoke-only`: it returns mock actions.
