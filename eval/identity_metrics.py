#!/usr/bin/env python3
"""Identity metrics counted from the ground truth object's side.

The existing SID in ``carla_sim/evaluate_prompt_metrics.py`` counts when a
*predicted track* switches between a prompt-valid and a prompt-invalid GT — it
measures semantic contamination of a track, and it is the right number for
"did the tracker start following the wrong car".

It is not the number for "how often did the same car get a new id". A tracker
that loses a target and re-acquires it under a fresh id, over and over, never
contaminates anything: each of its tracks matches only valid GT, so SID stays
near zero while identity is in fact shattered. On rmot3, with an erratically
moving drone camera, that is the dominant failure — one target across 300 frames
was covered by 23 different track ids while SID reported 8.

This module counts it from the GT side instead:

``id_reassignments``
    For each GT object, walk its frames in order and add 1 every time the
    predicted id matched to it differs from the previous one. Re-acquiring the
    same id after a gap costs nothing; a genuinely new id costs 1. This is the
    "+1 each time the same object is given a new id" count.

``fragments``
    Number of distinct predicted ids that ever matched this GT object. For a
    perfect tracker this is 1.

``id_purity``
    Fraction of the object's matched frames covered by its most common id — 1.0
    when one id covers the whole trajectory. Scale-free, so it compares across
    sequences of different length and visibility.

Reported per role (target / confuser / bus / distractor) because only the
target's identity is what the referring task is judged on.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def per_object_identity(gt_data: dict,
                        predictions: Dict[int, List[dict]],
                        iou_threshold: float = 0.5,
                        roles: Optional[Sequence[str]] = None) -> Dict[str, dict]:
    """Identity statistics per GT object, grouped by role.

    ``gt_data`` is the CARLA ``gt.json`` dict; ``predictions`` maps frame id to a
    list of ``{"track_id": int, "bbox_xyxy": [x1,y1,x2,y2]}``.

    Each GT frame is matched to the single best-IoU prediction above threshold,
    greedily and independently per frame — this measures identity continuity,
    not association quality, so a simple argmax is the right matcher here.
    """
    gt_by_frame: Dict[int, List[dict]] = defaultdict(list)
    for a in gt_data.get("annotations", []):
        gt_by_frame[a["image_id"]].append(a)

    # actor_id is stable across frames; gt_id is per-annotation.
    seq_by_actor: Dict[int, List[int]] = defaultdict(list)
    role_of: Dict[int, str] = {}

    for frame_id in sorted(gt_by_frame):
        preds = predictions.get(frame_id, [])
        for a in gt_by_frame[frame_id]:
            actor = a.get("actor_id", a.get("gt_id"))
            role_of[actor] = a.get("role", "?")
            if roles and a.get("role") not in roles:
                continue
            best, best_iou = None, iou_threshold
            for p in preds:
                v = _iou(p["bbox_xyxy"], a["bbox_xyxy"])
                if v >= best_iou:
                    best, best_iou = p, v
            if best is not None:
                seq_by_actor[actor].append(int(best["track_id"]))

    out: Dict[str, dict] = {}
    by_role: Dict[str, List[dict]] = defaultdict(list)
    for actor, ids in seq_by_actor.items():
        if not ids:
            continue
        reassign = sum(1 for i in range(1, len(ids)) if ids[i] != ids[i - 1])
        counts = Counter(ids)
        stat = {
            "actor_id": actor,
            "role": role_of.get(actor, "?"),
            "matched_frames": len(ids),
            "id_reassignments": reassign,
            "fragments": len(counts),
            "id_purity": counts.most_common(1)[0][1] / len(ids),
        }
        by_role[stat["role"]].append(stat)

    for role, rows in by_role.items():
        tot = sum(r["matched_frames"] for r in rows)
        out[role] = {
            "objects": len(rows),
            "matched_frames": tot,
            "id_reassignments": sum(r["id_reassignments"] for r in rows),
            "fragments": sum(r["fragments"] for r in rows),
            "id_purity": (sum(r["id_purity"] * r["matched_frames"] for r in rows) / tot)
                         if tot else 0.0,
            "objects_detail": rows,
        }
    return out


def format_report(per_role: Dict[str, dict], indent: str = "    ") -> str:
    """Human-readable block, target first."""
    if not per_role:
        return f"{indent}(no GT objects matched any prediction)"
    order = [r for r in ("target", "confuser", "bus", "distractor") if r in per_role]
    order += [r for r in sorted(per_role) if r not in order]
    lines = [f"{indent}{'role':11} {'objs':>5} {'matched':>8} "
             f"{'reassign':>9} {'frags':>6} {'purity':>7}"]
    for role in order:
        s = per_role[role]
        lines.append(f"{indent}{role:11} {s['objects']:5d} {s['matched_frames']:8d} "
                     f"{s['id_reassignments']:9d} {s['fragments']:6d} {s['id_purity']:7.3f}")
    return "\n".join(lines)


def load_predictions_mot(path: str) -> Dict[int, List[dict]]:
    """MOT-format file -> {frame_id: [{track_id, bbox_xyxy}]}, 0-indexed frames."""
    out: Dict[int, List[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            fr, tid = int(p[0]) - 1, int(p[1])
            x, y, w, h = (float(v) for v in p[2:6])
            out[fr].append({"track_id": tid, "bbox_xyxy": [x, y, x + w, y + h]})
    return out


def main() -> None:
    import argparse, glob, json, os
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="Run dir containing results/*.txt")
    ap.add_argument("--dataset", default="dataset/rmot3")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--role", default="target",
                    help="Role to summarise in the per-sequence table (default: target)")
    a = ap.parse_args()

    rows = []
    for txt in sorted(glob.glob(os.path.join(a.run, "results", "*.txt"))):
        seq = os.path.basename(txt)[:-4]
        gt_path = os.path.join(a.dataset, seq, "gt.json")
        if not os.path.isfile(gt_path):
            continue
        gt = json.load(open(gt_path))
        stats = per_object_identity(gt, load_predictions_mot(txt), a.iou)
        rows.append((seq, stats.get(a.role)))

    print(f"\nIdentity of the '{a.role}' — {a.run}")
    print(f"{'sequence':16} {'matched':>8} {'reassignments':>14} {'fragments':>10} {'purity':>8}")
    tr = tf = tm = 0
    for seq, s in rows:
        if not s:
            print(f"{seq:16} {'-':>8} {'-':>14} {'-':>10} {'-':>8}")
            continue
        print(f"{seq:16} {s['matched_frames']:8d} {s['id_reassignments']:14d} "
              f"{s['fragments']:10d} {s['id_purity']:8.3f}")
        tr += s["id_reassignments"]; tf += s["fragments"]; tm += s["matched_frames"]
    print(f"{'TOTAL':16} {tm:8d} {tr:14d} {tf:10d}")


if __name__ == "__main__":
    main()
