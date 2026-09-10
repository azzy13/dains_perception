#!/usr/bin/env python3
"""Side-by-side video: tracking frames on the left, the scene graph on the right.

The JSONL a run writes is not readable by eye, and the per-frame PNGs from
visualize_scene_graph.py draw the graph *on* the image, where the edges pile up
in the middle. This renders the graph as an actual graph (via scene_graph_dot ->
graphviz) and encodes the pair as one MP4, so the structure can be watched
evolving alongside the tracking.

    python eval/scene_graph_video.py \
        --run outputs/rmot3_viewerpref_0820 --seq cfg06_behind \
        --images dataset/rmot3/cfg06_behind/images

    # every sequence in a run
    python eval/scene_graph_video.py --run outputs/rmot3_viewerpref_0820 --all \
        --dataset dataset/rmot3

Needs the `dot` binary (graphviz) and ffmpeg on PATH.
"""
import argparse, json, os, subprocess, sys, tempfile, shutil
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_graph_dot import frame_graph_to_dot, render_dot

ROLE_COLOR = {"target_candidate": (0, 255, 0), "anchor": (0, 165, 255)}


def _draw_boxes(img, fg):
    """Boxes for this frame's graph nodes, coloured by role."""
    out = img.copy()
    for n in fg.get("nodes", []):
        x, y, w, h = n["bbox_tlwh"]
        c = ROLE_COLOR.get(n.get("role"), (200, 200, 200))
        p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
        cv2.rectangle(out, p1, p2, c, 2)
        cv2.putText(out, f"{n.get('role','?')[:6]} {n['track_id']}",
                    (p1[0], max(12, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return out


def _graph_png(fg, w, h):
    """Render one frame graph to a BGR image of exactly (h, w), letterboxed."""
    try:
        png = render_dot(frame_graph_to_dot(fg), fmt="png")
        arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        arr = None
    canvas = np.full((h, w, 3), 255, np.uint8)
    if arr is None or arr.size == 0:
        cv2.putText(canvas, "no graph", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 2)
        return canvas
    s = min(w / arr.shape[1], h / arr.shape[0], 1.0)
    arr = cv2.resize(arr, (max(1, int(arr.shape[1] * s)), max(1, int(arr.shape[0] * s))))
    y0, x0 = (h - arr.shape[0]) // 2, (w - arr.shape[1]) // 2
    canvas[y0:y0 + arr.shape[0], x0:x0 + arr.shape[1]] = arr
    return canvas


def build(jsonl, images_dir, out_mp4, fps=10, height=720, limit=None):
    frames = [json.loads(l) for l in open(jsonl)]
    if limit:
        frames = frames[:limit]
    if not frames:
        print(f"  no frames in {jsonl}"); return False

    tmp = tempfile.mkdtemp(prefix="sgvid_")
    try:
        n = 0
        for fg in frames:
            fid = fg["frame_id"]
            ip = os.path.join(images_dir, f"{fid:06d}.png")
            img = cv2.imread(ip)
            if img is None:
                continue
            scale = height / img.shape[0]
            left = cv2.resize(_draw_boxes(img, fg),
                              (int(img.shape[1] * scale), height))
            right = _graph_png(fg, left.shape[1] // 2, height)
            pair = np.hstack([left, right])
            cv2.putText(pair, f"frame {fid}   nodes={fg.get('num_tracks',0)} "
                              f"cand={fg.get('num_target_candidates',0)} "
                              f"anchor={fg.get('num_anchors',0)}",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(tmp, f"{n:06d}.png"), pair)
            n += 1
        if n == 0:
            print("  no frames rendered"); return False
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", os.path.join(tmp, "%06d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_mp4], check=True)
        print(f"  {out_mp4}  ({n} frames)")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="Run dir containing results/*.jsonl")
    ap.add_argument("--seq", help="Sequence name; omit with --all")
    ap.add_argument("--all", action="store_true", help="Every sequence in the run")
    ap.add_argument("--images", help="Images dir (single --seq)")
    ap.add_argument("--dataset", default="dataset/rmot3",
                    help="Dataset root, used with --all to find <seq>/images")
    ap.add_argument("--outdir", default=None, help="Default: <run>/graph_videos")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--limit", type=int, default=None, help="First N frames only")
    a = ap.parse_args()

    res = os.path.join(a.run, "results")
    outdir = a.outdir or os.path.join(a.run, "graph_videos")
    os.makedirs(outdir, exist_ok=True)

    if a.all:
        seqs = sorted(f[:-len("_scene_graphs.jsonl")] for f in os.listdir(res)
                      if f.endswith("_scene_graphs.jsonl"))
    else:
        if not a.seq:
            ap.error("--seq is required unless --all is given")
        seqs = [a.seq]

    for s in seqs:
        jsonl = os.path.join(res, f"{s}_scene_graphs.jsonl")
        images = a.images if (a.seq and a.images) else os.path.join(a.dataset, s, "images")
        if not os.path.isfile(jsonl):
            print(f"  skip {s}: no {jsonl}"); continue
        print(f"{s}:")
        build(jsonl, images, os.path.join(outdir, f"{s}_graph.mp4"),
              fps=a.fps, height=a.height, limit=a.limit)


if __name__ == "__main__":
    main()
