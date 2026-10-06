import numpy as np
import pytest

from src.bball.analytics.ball_labels import crop_with_label, is_ball_coloured


def _frame_with_patch(bgr, size=(720, 1280)):
    fr = np.full((*size, 3), (140, 190, 220), np.uint8)       # tan floor
    fr[300:320, 600:620] = bgr
    return fr


def test_orange_ball_passes_and_shoes_fail():
    orange = _frame_with_patch((40, 110, 200))                # BGR orange-brown
    purple = _frame_with_patch((200, 90, 160))                # purple shoe
    white = _frame_with_patch((245, 245, 245))                # white shoe
    yellow = _frame_with_patch((60, 230, 240))                # yellow shoe
    assert is_ball_coloured(orange, 610, 310, 20)
    assert not is_ball_coloured(purple, 610, 310, 20)
    assert not is_ball_coloured(white, 610, 310, 20)
    assert not is_ball_coloured(yellow, 610, 310, 20)


def test_crop_label_points_at_the_ball():
    fr = np.zeros((720, 1280, 3), np.uint8)
    crop, label = crop_with_label(fr, 900, 400, 22, 22, rng=np.random.default_rng(1))
    cls, bx, by, bw, bh = label.split()
    assert crop.shape[:2] == (640, 640) and cls == "0"
    assert 0 < float(bx) < 1 and 0 < float(by) < 1
    assert float(bw) == pytest.approx(22 / 640)


def test_negative_crop_has_empty_label():
    _, label = crop_with_label(np.zeros((720, 1280, 3), np.uint8), 100, 100, 20, 20, with_ball=False)
    assert label == ""
