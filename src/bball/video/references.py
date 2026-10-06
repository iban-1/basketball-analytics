"""Reference court views: two calibrated by hand, more added automatically.

The hand-picked views A and B show each basket. Frames panning between them are registered to
the references; frames that register strongly AND whose refined court lines sit on the painted
lines become new references when they cover a part of the court the existing ones do not
(judged by which court X coordinate is under the image centre). This extends coverage toward
mid-court without more hand calibration, and each addition is checked against the painted lines.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.bball.video.court import L
from src.bball.video.homography import apply_h, refine_homography
from src.bball.video.register import RefView, _features, build_ref, register


def read_frame(cap: cv2.VideoCapture, n: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n))
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"cannot read frame {n}")
    return frame


def hand_refs(cap: cv2.VideoCapture, cfg: dict) -> list[RefView]:
    return [build_ref(r["name"], r["frame"], read_frame(cap, r["frame"]), r["picks"],
                      refine=r.get("refine", False)) for r in cfg["references"]]


def auto_refs(cap: cv2.VideoCapture, path: Path) -> list[RefView]:
    """Rebuild automatically added references from the saved homographies."""
    if not path.exists():
        return []
    out = []
    for item in json.loads(path.read_text()):
        frame = read_frame(cap, item["frame"])
        pts, desc = _features(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        out.append(RefView(item["name"], item["frame"], np.array(item["H_img_from_court"]), pts, desc, 0.0))
    return out


def centre_court_x(H_img_from_court: np.ndarray, shape=(720, 1280)) -> float:
    """Court X (metres) lying under the image centre."""
    c = apply_h(np.linalg.inv(H_img_from_court), np.array([[shape[1] / 2, shape[0] / 2]]))
    return float(c[0, 0])


def expand(cap: cv2.VideoCapture, refs: list[RefView], n_frames: int, step: int = 30,
           min_inliers: int = 60, min_score: float = 2.0, bin_m: float = 2.5, rounds: int = 3,
           ) -> list[dict]:
    """Add references for court regions the current set covers poorly. Returns the new ones."""
    added: list[dict] = []
    refs = list(refs)
    for rnd in range(rounds):
        covered = {round(centre_court_x(r.H_img_from_court) / bin_m) for r in refs}
        best: dict[int, tuple[float, int, np.ndarray]] = {}
        for n in range(0, n_frames, step):
            frame = read_frame(cap, n)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, _, inl = register(gray, refs, min_inliers=min_inliers)
            if H is None:
                continue
            H, score = refine_homography(gray, H, stages=((3.0, 12.0), (1.5, 6.0)), reg=2e-3)
            if not np.isfinite(score) or score < min_score:
                continue
            b = round(centre_court_x(H) / bin_m)
            if b in covered:
                continue
            quality = score * min(inl, 300)
            if b not in best or quality > best[b][0]:
                best[b] = (quality, n, H)
        if not best:
            break
        for b, (_, n, H) in best.items():
            name = f"auto{len(added)}"
            pts, desc = _features(cv2.cvtColor(read_frame(cap, n), cv2.COLOR_BGR2GRAY))
            refs.append(RefView(name, n, H, pts, desc, 0.0))
            added.append({"name": name, "frame": int(n), "H_img_from_court": H.tolist(),
                          "court_x_under_centre": round(centre_court_x(H), 1)})
    return added
