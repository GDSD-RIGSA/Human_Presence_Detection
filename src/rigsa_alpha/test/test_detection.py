"""Unit tests for the radar detector (no ROS needed): python3 -m pytest test/"""
import numpy as np

from rigsa_alpha.detection import BackgroundMask, Detector, dbscan


def empty_room():
    grid = np.zeros((200, 200), dtype=np.int8)   # 10 x 10 m at 5 cm, walls on the border
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 100
    return BackgroundMask(grid, 0.05, (0.0, 0.0))


def person_points(rng, x, y, n=6):
    return np.column_stack([rng.normal(x, 0.15, n), rng.normal(y, 0.15, n), rng.uniform(0.3, 1.6, n)])


def test_dbscan_finds_two_clusters_and_noise():
    rng = np.random.default_rng(0)
    xy = np.vstack([rng.normal(2, 0.1, (10, 2)), rng.normal(6, 0.1, (10, 2)), [[9.0, 0.5]]])
    labels = dbscan(xy)
    assert labels.max() == 1
    assert labels[-1] == -1


def test_wall_points_are_background():
    mask = empty_room()
    assert mask.is_background(np.array([[0.1, 5.0], [9.9, 5.0], [-3.0, 5.0]])).all()
    assert not mask.is_background(np.array([[5.0, 5.0]])).any()


def test_walking_person_confirmed_once():
    rng = np.random.default_rng(1)
    det = Detector(empty_room())
    alerts = []
    for k in range(40):
        t = k * 0.1
        wall = np.column_stack([np.full(8, 0.05), rng.uniform(1, 9, 8), rng.uniform(0, 2, 8)])
        pts = np.vstack([person_points(rng, 2.0 + 0.08 * k, 5.0), wall])
        alerts += det.update(pts, t)
    assert len(alerts) == 1
    assert abs(alerts[0].pos[1] - 5.0) < 0.5


def test_single_noise_points_raise_no_alert():
    rng = np.random.default_rng(2)
    det = Detector(empty_room())
    alerts = []
    for k in range(50):
        pts = np.column_stack([rng.uniform(1, 9, 2), rng.uniform(1, 9, 2), rng.uniform(0.3, 1.5, 2)])
        alerts += det.update(pts, k * 0.1)
    assert alerts == []
