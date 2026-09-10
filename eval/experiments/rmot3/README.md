# rmot3 diagnostic experiments

Standalone, read-only scripts backing every number in `eval/rmot3_findings.md`.
Run from the repo root in the `dino_real` env. Experiments 1, 3, 7, 8 are
CPU-only; 2 needs a GPU.

| script | question | headline result |
|---|---|---|
| `exp1_color.py` | colour accuracy vs GT colour values, all 11 seqs, percentile sweep | red 15.4% → 97.9% |
| `exp2_thresh.py` | why does the anchor go missing? sweeps box/text threshold | text 0.80 → 0.40 kills empty phrases; bus→anchor 15/15 |
| `exp3_size.py` | is the target physically detectable per sequence? | cfg03_front 2.2k px², cfg05_front vis 0.12 |
| `exp4_colorcls.py` | `color_classifier` peer-relative scorer vs `scene_graph` LAB bins | achromatic works, chromatic abstained |
| `exp5_unified.py` | does a chroma floor in `patch_votes` fix red? | no — wrong gate |
| `exp6_chromastat.py` | is the gate `lab_stats` whole-crop median chroma? | yes: 0.0% → 99.3% past CHROMA_MIN |
| `exp7_relation.py` | relation accuracy given PERFECT GT boxes | viewer-centric 97.3% WRONG |
| `exp8_signs.py` | is it a sign error, and is the geometry recoverable? | flip → 97.3% correct; 3D bound 99.2% |
| `exp9_heading.py` | is image-space heading a valid proxy for travel direction? | no — 50% aligned, 37% opposed |
| `exp10_egomotion.py` | does ego-motion compensation rescue it? | 50% → 63% aligned; still loses to 97.3% |

`exp2_thresh.py` takes a sequence name, e.g. `python3 eval/experiments/rmot3/exp2_thresh.py cfg04_behind`.

Note `exp5` is a recorded negative result — the obvious fix that did not work.
It is kept deliberately: it is what narrowed the cause to `lab_stats`.
