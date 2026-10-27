# Original experiment visuals

Created on 27 October 2026 for the repository overview. These graphics reorganize existing evidence; no training, inference evaluation, or result files were changed.

## Source and scope

- Source revision: [409aa8bde8455e742bc5f4d087437fedc5d1565c](https://github.com/leonyoon-3dai/behavior-submission-2026/tree/409aa8bde8455e742bc5f4d087437fedc5d1565c).
- Frames and GIFs show Leonard Yoon's final-panel simulation runs using BEHAVIOR/OmniGibson assets. The simulator, robot assets, Ilia base model and EXPO method are upstream work.
- The three source MP4s were downloaded and their SHA256 values matched [VIDEOS.md](../../../VIDEOS.md).
- Existing PNG frames retain their recorded pixels. Composites add labels and resize for display. Some panels crop to the head camera. No generated scene imagery or reconstructed robot motion is used.
- Charts use [results.csv](../../../results/results.csv) and [verified-facts.json](../../reports/submission-2026/presentation-assets/verified-facts.json).
- All chart axes show goal progress on the common 0-to-1 range. Goal progress is neither success rate nor an organizer-verified leaderboard score.
- See [provenance.json](provenance.json) for asset checksums and video excerpt details.

## Visual index

| Asset | Evidence |
|---|---|
| experiment-overview.jpg | Original frames from task001/public0 at 132 s, task002/public0 at 300 s, task005/public1 at 509.8 s |
| policy-interface.jpg | Original task001/public0 three-camera frame at 60 s; [contract.py](../../../submission/contract.py) and [server.py](../../../submission/server.py) |
| adaptation-and-execution.png | [Final inference audit](../../submission-2026/ILIA-RT-EXPO-INFERENCE-AUDIT.md) |
| final-per-instance.png | All 12 unmodified final result rows |
| development-comparison.png | Six-task, 36-rollout pilot, kept separate from the final protocol |
| task001-storyboard.jpg | Original three-camera frames at 60, 132 and 237 s; [failure analysis](../../submission-2026/task001-failure-analysis/README.md) |
| reproducibility.png | [Evaluation guide](../../../EVALUATION-README.md), final model manifest and original output archive |
| task001-public0-preview.gif | Head-camera excerpt, video 60–66 s |
| task002-public0-preview.gif | Head-camera excerpt, video 292–298 s |
| task005-public1-preview.gif | Head-camera excerpt, video 100–106 s |

Video timestamps refer to the MP4 playback timeline, not wall-clock runtime. GIFs sample at 2 FPS, resize to 224 pixels and quantize colors to reduce README download size. They are previews; full-resolution MP4s remain unchanged.

The task001 frame sequence supports visual review of movement and late repeated poses. It does not establish precise contact state or a causal explanation for failure. Final RT-only runs have no matched A0 comparison.

## Rebuild

Requires Python with Pillow, NumPy and Matplotlib, the DejaVu Sans font, and FFmpeg. The script downloads pinned source images and the three checksum-verified original videos. It caches these in local dot-directories and rewrites only the overview assets in this folder.

```bash
python docs/assets/experiment-gallery-20261027/build_visuals.py
```

Original frame provenance:
[rollout-source-provenance.json](../../reports/submission-2026/presentation-assets/rollout-source-provenance.json).

