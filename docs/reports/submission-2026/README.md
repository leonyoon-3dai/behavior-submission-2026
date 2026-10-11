# Visual research reports

By **Leonard Yoon**, Independent researcher. Actual robot-camera frames, measured results, and concise explanations of Ilia + EXPO-RT.

| Read | Contents |
|---|---|
| [Presentation PDF](presentation-pilot6-submission12.pdf) | 12 slides: tasks, method, results, observed failures, and next steps |
| [Research paper](research-paper-pilot6-submission12.pdf) | Technical analysis, figures, and a dataset/robot appendix |
| [LaTeX source](research-paper-pilot6-submission12.tex) | Editable manuscript and references |

**Two separate panels:** 36 development rollouts compare original Ilia with RT; 12 final rollouts evaluate RT on three tasks. No new rollouts were run for this visual revision. The pilot did not show an overall improvement, and the final panel had no complete successes.

**Status:** the challenge portal accepted the submission on October 11, 2026; organizer verification is pending. The paper is a research draft, not submitted to arXiv.

## Sources and image provenance

- [Frozen results snapshot](results-snapshot.json) and [final metrics](../../../results/results.json). The snapshot retains its original pre-submission status; current status is in the [project overview](../../../README.md).
- [Presentation provenance](presentation-assets/rollout-source-provenance.json) and [paper provenance](paper-assets/provenance.json): source videos, checksums, timestamps, and crops. Frames illustrate behavior; original videos and metric files remain unchanged.
- Demonstration crops come from the earlier presentation at [source commit `4d40b4a`](https://github.com/leonyoon-3dai/behavior-submission-2026/tree/4d40b4a02f7f9042449353ac96f9574a08a909b7). The robot illustration is attributed to BEHAVIOR / Stanford Vision and Learning Lab in the paper.
- [Manifest and file hashes](manifest.json) identify the sources, builders, and artifacts in this revision.

## Rebuild the reports

From the repository root, use Python with ReportLab, Pillow, Matplotlib, and NumPy, plus DejaVu Sans fonts and a LaTeX installation with the manuscript's packages. The [original Python dependency lock](requirements.lock.txt) is preserved for reference.

```bash
python scripts/build_visual_presentation.py
python docs/reports/submission-2026/paper-assets/build_figures.py
cd docs/reports/submission-2026
pdflatex -interaction=nonstopmode -halt-on-error research-paper-pilot6-submission12.tex
pdflatex -interaction=nonstopmode -halt-on-error research-paper-pilot6-submission12.tex
```

Rebuilding uses the included frame assets; no GPU or simulator is needed. PDF metadata can change between builds, so hashes identify this published revision rather than guaranteeing byte-identical rebuilds.

[Evaluation guide](../../../EVALUATION-README.md) · [Original rollout videos](../../../VIDEOS.md) · [Project overview](../../../README.md)
