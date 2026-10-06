"""Grow the set of court reference views automatically (run once per clip).

Run:  python -m scripts.build_references
Writes results/auto_references.json (frame numbers and homographies only, no image data).
"""
from __future__ import annotations

import json

import cv2

from src.bball.config import load_config, resolve
from src.bball.video.references import centre_court_x, expand, hand_refs


def main() -> None:
    cfg = load_config()["video"]
    cap = cv2.VideoCapture(str(resolve(cfg["path"])))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    refs = hand_refs(cap, cfg)
    print("hand references:", [(r.name, r.frame_no, f"court X under image centre = {centre_court_x(r.H_img_from_court):.1f} m")
                               for r in refs])
    added = expand(cap, refs, n_frames, min_inliers=cfg["min_inliers"] + 10)
    out = resolve(load_config()["paths"]["results"]) / "auto_references.json"
    out.write_text(json.dumps(added, indent=1), encoding="utf-8")
    print(f"{len(added)} references added:")
    for a in added:
        print("  ", a["name"], "frame", a["frame"], "court X under centre", a["court_x_under_centre"], "m")


if __name__ == "__main__":
    main()
