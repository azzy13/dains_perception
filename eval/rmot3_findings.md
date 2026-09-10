# rmot3 — bugs found, what each changed

`dataset/rmot3`: 11 seqs × 300 frames, CARLA drone cam. 7× "red sedan behind the
bus", 4× "in front of". Matched confuser on the opposite side, so only the
relation can answer. Measured 2026-09-10. Scripts: `eval/experiments/rmot3/`.

## Change → effect

| # | change | file | effect |
|---|---|---|---|
| 1 | relation sign inverted | `scene_graph.py` `_depth_relation` | **1.5% → 97.3%** correct (GT boxes) |
| 2 | `text_threshold` 0.80→0.40 | CLI only | empty phrases **77% → 0%**; cfg03 anchors **16→269**/300 |
| 3 | colour sampled over background | `scene_graph.py` `_dominant_color_from_crop` | red **15.4% → 97.9%**, bus-as-red 0.0% |
| 4 | same bug in chroma gate | `color_classifier.py` `lab_stats` | red past `CHROMA_MIN` **0% → 99.3%**; margin +0.002 → **+0.514** |
| 5 | tensor on wrong GPU | `worker_clean.py:965` +2 | GPU-1 silently returned **zero detections** |
| 6 | no `set_device()` per thread | `eval_carla.py` | `ms_deform_attn` on wrong device → device-side assert |
| 7 | `sys.path.pop(0)` race | `groundingdino/util/slconfig.py` | killed a GPU thread at load; now 48 concurrent loads, 0 errors |
| 8 | viz not disableable | `eval_carla.py` | added `--no-visualize_scene_graph` (~6 min/seq vs <1 min GPU) |

Originals of all edited files backed up. Nothing deleted.

## The main finding — relation answered backwards

GT boxes only, no detector/tracker/colour. 4,588 pairs.

| method | correct | wrong |
|---|---:|---:|
| GT 3D world-yaw (upper bound) | **99.2%** | 0.8% |
| viewer-centric, as written | **1.5%** | **97.3%** |
| viewer-centric, sign fixed | **97.3%** | 1.5% |
| object-centric, real GT heading | **43.9%** | 33.9% |

- Dataset is sound (99.2% bound).
- "In front of" = ahead along travel = **further** from a following camera, not
  nearer. Holds for drone + forward-facing car cams; reverses for oncoming views.
- **Object-centric is at chance, and the code prefers it when a heading exists** —
  picks 44% over 97%. This is the research problem, not a bug.

## Why `front` collapsed (baseline, 11/11)

| group | SP | SR | DCR |
|---|---:|---:|---:|
| behind (7) | 0.42–0.83 | 0.29–0.93 | 0.17–0.53 |
| front (4) | **0.000–0.086** | 0.000–0.153 | **0.86–0.996** |

DCR 0.994 = nearly every box was the confuser. ~100% wrong = inversion; chance = 50%.

## Heading recovery fails under a moving camera (drone)

Reference "ahead" direction from GT (in a `behind` clip the confuser leads and
the target trails). Compared against what a tracker actually recovers — the
anchor's image-space displacement (`exp9`, `exp10`).

| anchor heading estimate | aligned <60° | opposed >120° | median angle |
|---|---:|---:|---:|
| raw image motion | 50% | **37%** | 61° |
| ego-motion compensated (minus frame median) | **63%** | 28% | 29° |

Image displacement = object motion **+** camera motion. With a flying drone the
camera term dominates, so the recovered heading is near-random — cfg10_behind is
62% *opposed* (median 136°). That is the whole explanation for the object-centric
branch scoring 43.9%.

Ego-compensation helps a lot where the scene is mostly static (cfg03 60→90%
aligned, cfg05 47→70%) and **hurts** where moving traffic dominates the median
(cfg08 30→7%, cfg10 14→30% but still 62% opposed). Not reliable enough: 63%
aligned still loses badly to 97.3%.

**Consequence:** `_depth_relation` preferred the heading branch whenever a
heading existed, so fixing anchor detection *increased* how often the worse
estimator ran. `PREFER_VIEWER_DEPTH = True` now prefers image depth-order; set
False for static cameras or when ego-motion is compensated upstream.

**For the paper:** this is a clean negative result — object-centric spatial
relations are not recoverable from monocular image-space tracking under a moving
camera without pose or robust ego-motion estimation, and a viewer-centric depth
heuristic beats it decisively when the camera holds a consistent along-track
viewpoint.

## Colour — LAB was not the problem

Only 11–20% of a target box is car (vis 0.37–0.55); red pixels exist (chroma 50)
but average to gray. Same bug in 3 places.

**Do not swap in segmentation + a colour model:** GrabCut scored 11/27 vs 26/27
for a chroma threshold, and failed on 8/53 crops. `color_classifier.py` already
has the transfer-safe design; it was fed a broken measurement. `CHROMA_MIN=18`
needs no retuning (separates 99.3% from 5.3%).

**Refer-KITTI (untested prediction):** that module records "0.0% of red-annotated
crops reach CHROMA_MIN" — same bug, so fix #4 should carry. Its vocabulary is
achromatic-heavy (black 120, light/color 114/110, silver 66, red 36, white 10),
so rely on the peer-relative *scorer*; `black`/`white`/`silver` hard labels are
still 0%.

## Difficulty — quote per sequence, not the mean

| seq | med target area | med vis |
|---|---:|---:|
| cfg03_front | **2,196 px²** | 0.49 (30% <32px wide) |
| cfg13_front | 3,008 | 0.57 |
| cfg01_front | 4,180 | 0.46 |
| cfg05_front | 4,286 | **0.12** (57% frames <0.3) |
| cfg10_behind | 6,786 | 0.36 |
| others (6) | 6,676–20,468 | 0.42–0.65 |

`front` targets are systematically smaller → an aggregate reads a **difficulty
gap as a relational one**.

## Runs

| dir | valid? |
|---|---|
| `..._0709` | NO — crashed frame 1, printed all-zeros, exit 0 |
| `..._0712` | NO — 4/11 survived, averaged only those |
| `..._0722` | **YES, baseline 11/11** — SP 0.355 / SR 0.387 / DCR 0.608 / SID 9 |
| `rmot3_fixed_0803` | NO — slconfig race killed GPU-0; 5/11, front-skewed |
| `rmot3_allfixes_0808` | all fixes |

> **Trap:** `eval_carla.py` catches per-scenario exceptions and averages the
> survivors — a half-crashed run prints a summary that looks real. **Check
> `results/*.txt` == 11 before quoting anything.**

## Open

1. Object-centric relation from monocular tracking — 43.9% with perfect headings.
2. `_depth_relation` prefers object-centric over the better fallback. Untested: flip preference, or gate on heading reliability.
3. Identity: 23 track IDs for one target/300 frames (ByteTrack, no appearance). SP/SR/DCR are per-frame IoU-matched — identity affects **only SID**.
4. `query_grounding.py:190` sends 0–0 ties to `ROLE_TARGET`. Latent only (0% empty phrases at 0.40). Untouched — Week-3 schema is frozen.
5. No CARLA/rmot3 Optuna tuner exists. `text_threshold` is the highest-impact param.

## Repro

```bash
python3 eval/eval_carla.py --carla_scenarios dataset/rmot3 \
    --worker clean --grounded --fp16 --devices 0,1 --save_video \
    --text_threshold 0.40 --no-visualize_scene_graph --outdir outputs/<name>
```

`--worker clean` required: `--grounded` hard-errors on `simple`
(`eval_carla.py:565`), which has no query grounding at all.
