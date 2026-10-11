# robogoomy · Ilia + EXPO-RT

Leonard Yoon · Independent · BEHAVIOR-1K 2026

Frozen Ilia actor with task-specific EXPO-RT candidate selection and residual action editing. Final self-evaluation: **tasks 002 → 001 → 005, public 0–3 once each, 12 original rollouts**. This is a partial submission.

[Evaluation instructions](EVALUATION-README.md) · [Results ZIP](https://huggingface.co/leonayoon/behavior-submission-task001-002-005-public0-3-2026/resolve/main/package/submission12-final.zip?download=true) · [All 12 videos](VIDEOS.md) · [Public artifacts](https://huggingface.co/leonayoon/behavior-submission-task001-002-005-public0-3-2026)

| Task | Public 0 | Public 1 | Public 2 | Public 3 | Mean final Q |
|---|---:|---:|---:|---:|---:|
| 002 | 0.5714 | 0.7143 | 0.5714 | 0.0000 | 0.4643 |
| 001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 005 | 0.0000 | 0.5000 | 0.0000 | 0.3333 | 0.2083 |

Observed 12-rollout mean final Q: **0.224206**; full success: **0/12**. Missing instances count as zero: selected 3-task/30-instance mean **0.089683**, full 100-task/1000-instance mean **0.00269048**. The observed-panel mean is not the full leaderboard score.

![Final 12 results](docs/reports/submission-2026/submission12-status.png)

[Presentation PDF](docs/reports/submission-2026/presentation-pilot6-submission12.pdf) · [Research manuscript draft](docs/reports/submission-2026/research-paper-pilot6-submission12.pdf) · [Inference audit](docs/submission-2026/ILIA-RT-EXPO-INFERENCE-AUDIT.md)

The six-task development pilot contains 36 paired A0/RT rollouts on tasks 000,001,002,003,005,007. It is separate from the final 12-rollout result panel. The frozen image retains five task routes (000,001,002,003,005); task007 is pilot only. No extra final rollouts or favorable-repeat selection were performed. Public index0 had prior development exposure, which is disclosed in the evaluation README.

Original metrics and MP4 files are unchanged. A real 24GB hardware test was intentionally not performed; the deployment check uses the current L40S. Portal receipt is recorded separately once accepted. arXiv publication has not been performed.
