"""Court model, homography fitting/refinement and feature registration on synthetic images."""
import cv2
import numpy as np
import pytest

from src.bball.video.court import (ARC_R, CY, FT_RADIUS, HOOP_X, L, LANE_DEPTH, LANE_HALF, W,
                                   court_lines, landmark, sample_points)
from src.bball.video.homography import (alignment_score, apply_h, fit_homography, refine_homography)
from src.bball.video.register import build_ref, register


def _true_h():
    """A plausible court(m) -> pixel homography for a high sideline camera."""
    court = np.float32([[2, 2], [26, 2], [26, 13], [2, 13]])
    pix = np.float32([[120, 600], [1180, 600], [1000, 260], [260, 260]])
    return cv2.getPerspectiveTransform(court, pix).astype(np.float64)


def _paint_court(H, shape=(720, 1280)):
    img = np.full(shape, 90, np.uint8)
    for line in court_lines():
        p = apply_h(H, line)
        cv2.polylines(img, [p.astype(np.int32)], False, 255, 2, cv2.LINE_AA)
    return img


def test_official_nba_dimensions():
    assert (L, W) == (pytest.approx(28.65, abs=0.01), pytest.approx(15.24, abs=0.01))
    assert 2 * LANE_HALF == pytest.approx(4.877, abs=0.01)           # 16 ft lane
    assert LANE_DEPTH == pytest.approx(5.791, abs=0.01)              # free-throw line 19 ft from baseline
    assert FT_RADIUS == pytest.approx(1.829, abs=0.01) and ARC_R == pytest.approx(7.239, abs=0.01)
    assert HOOP_X == pytest.approx(1.6, abs=0.03)


def test_landmarks_and_markings_are_symmetric_about_the_half_line():
    a, b = landmark("L ft near corner"), landmark("R ft near corner")
    assert a[0] + b[0] == pytest.approx(L) and a[1] == pytest.approx(b[1]) == pytest.approx(CY - LANE_HALF)
    pts = sample_points(0.2)
    assert pts[:, 0].min() >= -1e-6 and pts[:, 0].max() <= L + 1e-6 and pts[:, 1].max() <= W + 1e-6


def test_fit_homography_recovers_the_mapping():
    H = _true_h()
    court = np.array([[3, 3], [25, 4], [24, 12], [4, 11], [14, 7]], float)
    Hf, resid = fit_homography(court, apply_h(H, court))
    assert resid.max() < 1e-4
    assert np.allclose(apply_h(Hf, [[10.0, 5.0]]), apply_h(H, [[10.0, 5.0]]), atol=1e-4)


def test_refinement_pulls_a_misplaced_court_back_onto_the_lines():
    H = _true_h()
    img = _paint_court(H)
    anchors = np.array([[7, 4], [21, 4], [21, 11], [7, 11], [14, 7.6]], float)
    truth = apply_h(H, anchors)
    shift = np.float64([[1, 0, 9], [0, 1, -7], [0, 0, 1]])           # 9 px right, 7 px up
    H_bad = shift @ H
    err_before = np.linalg.norm(apply_h(H_bad, anchors) - truth, axis=1).mean()
    H_ref, score = refine_homography(img, H_bad)
    err_after = np.linalg.norm(apply_h(H_ref, anchors) - truth, axis=1).mean()
    assert err_before > 8 and err_after < 2.5 and err_after < err_before / 3
    far_off = shift @ shift @ shift @ shift @ H                       # 36 px right, 28 px up
    assert alignment_score(img, H_ref) > 1.5 * alignment_score(img, far_off)


def test_registration_recovers_a_pan_and_zoom_from_features():
    """Warp a textured image (like a pan + zoom) and recover the mapping with SIFT + RANSAC."""
    rng = np.random.default_rng(0)
    base = (rng.random((720, 1280)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
    H_court_to_ref = _true_h()
    picks = {"L baseline near lane": apply_h(H_court_to_ref, [landmark("L baseline near lane")])[0].tolist(),
             "L baseline far lane": apply_h(H_court_to_ref, [landmark("L baseline far lane")])[0].tolist(),
             "L ft near corner": apply_h(H_court_to_ref, [landmark("L ft near corner")])[0].tolist(),
             "L ft far corner": apply_h(H_court_to_ref, [landmark("L ft far corner")])[0].tolist()}
    ref = build_ref("T", 0, cv2.cvtColor(base, cv2.COLOR_GRAY2BGR), picks)
    move = np.float64([[1.15, 0.02, -70], [0.01, 1.15, -40], [0, 0, 1]])        # zoom 15 % + pan
    moved = cv2.warpPerspective(base, move, (1280, 720))
    H, name, inliers = register(moved, [ref], min_inliers=15)
    assert name == "T" and inliers >= 30
    probe = np.array([[14.0, 7.6]])
    assert np.linalg.norm(apply_h(H, probe) - apply_h(move @ H_court_to_ref, probe)) < 3.0
