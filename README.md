# robogoomy | BEHAVIOR 2026

**Ilia + EXPO-RT — learning small action corrections for household robots.**

Leonard Yoon · Independent researcher

Can small learned corrections help a pretrained robot policy complete household tasks? We test Ilia's released policy with EXPO-RT in simulated homes and share the results, videos, and code here.

> **Submitted October 11, 2026 · Under verification.** Portal acceptance is confirmed; organizer performance verification is pending.

[Run the policy](EVALUATION-README.md) · [Watch the rollouts](VIDEOS.md) · [Presentation](docs/reports/submission-2026/presentation-pilot6-submission12.pdf) · [Research paper](docs/reports/submission-2026/research-paper-pilot6-submission12.pdf)

[Download results](RESULTS.md) — Preview the package before downloading.

## Final results

**3 tasks · 4 public instances each · 12 rollouts.** One run per instance; all results and original videos are included.

| Task | Mean final Q | Fully completed |
|---|---:|---:|
| 002 · Put away Halloween decorations | 0.4643 | 0/4 |
| 001 · Pick up trash | 0.0000 | 0/4 |
| 005 · Set mousetraps | 0.2083 | 0/4 |

**Q is the fraction of goals satisfied at episode end.** Mean Q was **0.224206**, with partial progress but **0/12 complete successes**.

![Final goal progress for each of the 12 rollouts](docs/reports/submission-2026/submission12-status.png)

**Self-reported:** counting unsubmitted instances as zero gives **0.00269048** over the full 1,000-instance challenge. The 12-rollout mean is not the leaderboard score. [Per-instance results](results/results.csv)

## The approach

1. **Propose:** Ilia generates two action sequences.
2. **Refine:** a small network adds one edited alternative.
3. **Choose:** learned value networks select a candidate; Ilia's stage tracking and correction rules guide execution.

![Frozen Ilia policy with EXPO-RT action editing and selection](docs/reports/submission-2026/policy-pipeline.png)

Ilia's weights stay frozen. Task-specific networks train on **20 official demonstrations for 1,000 updates**. [Implementation](docs/submission-2026/ILIA-RT-EXPO-INFERENCE-AUDIT.md)

## What we learned

In a separate **6-task, 36-rollout pilot**, adding RT to the original policy (A0) reduced mean Q from **0.455 to 0.329** and successes from **5/18 to 1/18**. RT did not improve overall performance.

![Original policy versus action editing in the six-task development pilot](docs/reports/submission-2026/pilot6-Q.png)

The final evaluation used RT only, without a matched A0 comparison. Placement, coordination, and recovery are priorities for future investigation.

[Trash-picking analysis](docs/submission-2026/task001-failure-analysis/README.md) · [Instance comparison](docs/submission-2026/public3-analysis/README.md)

## Reproduce and explore

| Resource | Contents |
|---|---|
| [Evaluation guide](EVALUATION-README.md) | Docker, commands, wrapper, and robot config |
| [Policy source](submission/README.md) | Server interface and task routing |
| [Hugging Face archive](https://huggingface.co/leonayoon/behavior-submission-task001-002-005-public0-3-2026) | Results, videos, image archive, and reports |
| [Video index](VIDEOS.md) | 12 original videos and checksums |
| [Research draft](docs/reports/submission-2026/research-paper-pilot6-submission12.pdf) · [LaTeX](docs/reports/submission-2026/research-paper-pilot6-submission12.tex) | Method and analysis; not published to arXiv |

Docker includes the weights; no Hugging Face token is needed. The simulator runs separately. L40S deployment checks passed; **actual 24GB hardware remains untested**.

<details>
<summary>Evaluation scope and provenance</summary>

- Final order: 002, 001, 005; public indices 0–3, one rollout each.
- Image routes: 000, 001, 002, 003, 005. Task 007 was pilot-only.
- Public index 0 had development exposure; it is not an untouched test.
- Pilot results are separate. No statistical significance claim is made.
- Official Q differs from critic predictions. Intermediate Q maxima and complete goal traces were not captured.
- The three-task, 30-instance mean, including missing zeros, is 0.089683.
- Original results and videos remain unchanged; revisions and SHA256 checksums preserve their identities.
- Submission ID: `20261011T111517Z-robogoomy-ilia-expo-rt`.

</details>
