"""Unit tests for the radar detector (no ROS needed): python3 -m pytest test/"""
import numpy as np
import pytest

from rigsa_alpha.detection import (MAX_SPEED, BackgroundMask, Detector, assign, clamp_speed,
                                   dbscan, merge_close)


def empty_room():
    grid = np.zeros((200, 200), dtype=np.int8)   # 10 x 10 m at 5 cm, walls on the border
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 100
    return BackgroundMask(grid, 0.05, (0.0, 0.0))


def person_points(rng, x, y, n=6):
    return np.column_stack([rng.normal(x, 0.15, n), rng.normal(y, 0.15, n),
                            rng.uniform(0.3, 1.6, n)])


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
        pts = np.column_stack([rng.uniform(1, 9, 2), rng.uniform(1, 9, 2),
                               rng.uniform(0.3, 1.5, 2)])
        alerts += det.update(pts, k * 0.1)
    assert alerts == []


def test_dbscan_empty_input_returns_no_labels():
    assert len(dbscan(np.zeros((0, 2)))) == 0


def test_dbscan_min_points_boundary():
    same = np.zeros((5, 2))                       # 5 points on one spot = MIN_POINTS
    assert (dbscan(same) == 0).all()              # at the limit: a cluster
    assert (dbscan(same[:4]) == -1).all()         # one below: noise


def test_dbscan_eps_boundary_is_inclusive():
    on_eps = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert (dbscan(on_eps, eps=1.0, min_points=2) == 0).all()          # distance == eps
    beyond = np.array([[0.0, 0.0], [1.0000001, 0.0]])
    assert (dbscan(beyond, eps=1.0, min_points=2) == -1).all()         # just above eps


def test_dbscan_nan_points_are_noise():
    xy = np.vstack([np.zeros((5, 2)), [[np.nan, np.nan]]])
    labels = dbscan(xy)
    assert (labels[:5] == 0).all() and labels[5] == -1


def test_background_margin_boundary():
    mask = empty_room()   # wall on the border, margin 0.35 m
    assert mask.is_background(np.array([[5.0, 0.39]])).all()      # just inside the margin
    assert not mask.is_background(np.array([[5.0, 0.41]])).any()  # just outside it


def test_merge_close_boundary_is_strict():
    pair = [(np.array([0.0, 0.0]), 1), (np.array([1.0, 0.0]), 1)]
    assert len(merge_close(pair, dist=1.0)) == 2                  # distance == dist: kept apart
    merged = merge_close(pair, dist=1.0001)
    assert len(merged) == 1
    np.testing.assert_allclose(merged[0][0], [0.5, 0.0])


def test_merge_close_empty_and_single():
    assert merge_close([], dist=1.0) == []
    one = [(np.array([2.0, 2.0]), 3)]
    assert merge_close(one, dist=1.0) == one


def test_assign_big_cost_is_not_assigned():
    assert assign(np.array([[0.99]]), big=1.0) == [(0, 0)]        # just below big
    assert assign(np.array([[1.0]]), big=1.0) == []               # at big: rejected
    assert assign(np.zeros((0, 0)), big=1.0) == []                # nothing to assign


def test_clamp_speed_boundary():
    at_limit = np.array([MAX_SPEED, 0.0])
    np.testing.assert_array_equal(clamp_speed(at_limit), at_limit)
    fast = clamp_speed(np.array([3.0 * MAX_SPEED, 0.0]))
    assert np.linalg.norm(fast) == pytest.approx(MAX_SPEED)
