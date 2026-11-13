"""Score a run report (JSONL of job records, from `mhprep run --report` or py_baseline.py)
against corpus ground truth.

    python bench/evaluate.py corpus/test/truth.json results/cpp_report.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

TOLERANCE = 2  # frames; stride-2 analysis reports a cut at the first analyzed frame after it


def load_report(path: str) -> dict[str, dict]:
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            out[r["input"]] = r
    return out


def match_cuts(truth: list, pred: list[int], tol: int = TOLERANCE) -> int:
    """Greedy one-to-one matching within tolerance; returns true positives.

    A truth entry is a frame (hard cut) or a [first, last] frame interval (a dissolve or
    fade): a prediction anywhere in the interval, give or take `tol`, matches it."""
    return len(match_cut_pairs(truth, pred, tol))


def match_cut_pairs(truth: list, pred: list[int], tol: int = TOLERANCE) -> list[tuple[int, int]]:
    used, pairs = set(), []
    for ti, t in enumerate(truth):
        lo, hi = (t, t) if isinstance(t, int) else t
        dist = lambda p: max(lo - p, p - hi, 0)  # noqa: E731
        best = None
        for i, p in enumerate(pred):
            if i not in used and dist(p) <= tol and (best is None or dist(p) < dist(pred[best])):
                best = i
        if best is not None:
            used.add(best)
            pairs.append((ti, best))
    return pairs


def lineage(src: str) -> str | None:
    """Images that come from the same Magic Hour generation: a still and the clip animated
    from it, or several takes rendered from one start frame. A match between two members
    of a lineage is neither rewarded nor penalised (it is arguably a true duplicate)."""
    m = re.search(r"time-machine_(?:scenes|clips)_(\d\d)", src)
    if m:
        return "book" + m.group(1)
    if "strava" in src and ("finish" in src or "mile20" in src):
        return ("strava_finish_" if "finish" in src else "strava_mile20_") + \
            ("female" if "female" in src else "male")
    if "mitosis" in src and "shot0" in src:
        return "mitosis_shot0"
    if "outfit-planner_looks_clips" in src or "fit-check" in src:
        return "outfit_clip"
    return None


def segment_of(frame: int, video: dict) -> dict:
    for s in reversed(video["segments"]):
        if frame >= s["start_frame"]:
            return s
    return video["segments"][0]


def score(truth: dict, report: dict[str, dict]) -> dict:
    # --- scene cuts ---
    tp = fp = fn = 0
    by_kind = defaultdict(lambda: [0, 0])  # truth kind -> [found, total]
    kf_dup_tp = kf_dup_fp = kf_dup_fn = 0
    missing = 0
    for v in truth["videos"]:
        r = report.get(v["video"])
        if not r or r.get("status") != "succeeded":
            missing += 1
            fn += len(v["cuts"])
            continue
        m = r["manifest"]
        pred = [s["start_frame"] for s in m["scenes"][1:]]
        truth_cuts = v.get("cut_intervals", v["cuts"])
        pairs = match_cut_pairs(truth_cuts, pred)
        t = len(pairs)
        tp += t
        fp += len(pred) - t
        fn += len(truth_cuts) - t
        kinds = {tr["start"]: tr["kind"] for tr in v.get("transitions", [])}
        found = {ti for ti, _ in pairs}
        for ti, c in enumerate(truth_cuts):
            k = "cut" if isinstance(c, int) or c[0] == c[1] else kinds.get(c[0], "gradual")
            by_kind[k][1] += 1
            by_kind[k][0] += ti in found
        # --- keyframe dedup within a video ---
        assets = m["assets"]
        seen_groups = set()
        for s in m["scenes"]:
            if s["keyframe"] < 0:
                continue
            a = assets[s["keyframe"]]
            g = tuple(segment_of(a["frame"], v)["key"])  # same continuous source shot
            if "duplicate_of" in a:
                og = tuple(segment_of(assets[a["duplicate_of"]]["frame"], v)["key"])
                if og == g:
                    kf_dup_tp += 1
                else:
                    kf_dup_fp += 1
            elif g in seen_groups:
                kf_dup_fn += 1  # same shot seen before but we kept it
            seen_groups.add(g)

    # --- image dedup across uploads (one job per image, same tenant) ---
    img_truth = {i["path"]: i for i in truth["images"]}
    groups = defaultdict(list)
    for i in truth["images"]:
        groups[i["group"]].append(i["path"])
    job_by_id = {r["id"]: r for r in report.values()}
    expected_dups = sum(len(g) - 1 for g in groups.values())
    d_tp = d_fp = d_lineage = 0
    flagged = set()
    for path, it in img_truth.items():
        r = report.get(path)
        if not r or r.get("status") != "succeeded":
            continue
        a = r["manifest"]["assets"][0]
        if "cross_duplicate_of" in a:
            owner = job_by_id[a["cross_duplicate_of"]["job"]]["input"]
            if img_truth[owner]["group"] == it["group"]:
                d_tp += 1
                flagged.add(path)
            elif lineage(it["src"]) and lineage(it["src"]) == lineage(img_truth[owner]["src"]):
                d_lineage += 1
            else:
                d_fp += 1
    n_images = len(img_truth)
    images_ok = sum(1 for p in img_truth if report.get(p, {}).get("status") == "succeeded")
    flagged_total = d_tp + d_fp + d_lineage

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 1.0
        rc = tp_ / (tp_ + fn_) if tp_ + fn_ else 1.0
        return {"precision": round(p, 4), "recall": round(rc, 4),
                "f1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0,
                "tp": tp_, "fp": fp_, "fn": fn_}

    return {
        "videos_missing": missing,
        "scene_cuts": prf(tp, fp, fn),
        "scene_cut_recall_by_kind": {k: {"found": f, "total": n, "recall": round(f / n, 4)}
                                     for k, (f, n) in sorted(by_kind.items())},
        "keyframe_dedup": prf(kf_dup_tp, kf_dup_fp, kf_dup_fn),
        "image_dedup": {**prf(d_tp, d_fp, expected_dups - d_tp),
                        "images": n_images, "images_processed": images_ok,
                        "flagged": flagged_total, "flagged_same_generation": d_lineage,
                        "api_calls_avoided_pct": round(100 * flagged_total / max(1, n_images), 2)},
    }


if __name__ == "__main__":
    print(json.dumps(score(json.load(open(sys.argv[1])), load_report(sys.argv[2])), indent=2))
