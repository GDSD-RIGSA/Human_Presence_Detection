"""Unit tests for the room-mode detector options (no ROS needed): python3 -m pytest test/"""
import numpy as np
import pytest

from rigsa_alpha.detection import (BackgroundMask, Detector, Track, assign,
                                   merge_close)

ROOM = dict(gate=2.0, lost_after=4.0, reacquire_s=15.0, reacquire_m=3.0, eps=0.6)
IMPROVED = dict(ROOM, max_range=6.5, snr_db=16.0, snr_db_per_m=0.5, doppler_k=1.0, absorb_m=1.2,
                new_track_min_points=10, tracker='kalman', static_hold_s=30.0)
RADAR = (0.0, 0.0)


def open_room():
    return BackgroundMask(np.zeros((320, 280), dtype=np.int8), 0.05, (-2.0, -8.0))


def person(rng, x, y, vr, n=4, db=22.0):
    """n radar points of one person at (x, y) with radial velocity vr, as (N, 5) map points."""
    return np.column_stack([rng.normal(x, 0.12, n), rng.normal(y, 0.12, n), np.full(n, 1.0),
                            np.full(n, db), np.full(n, vr)])


def run(det, frames, dt=0.1):
    alerts = []
    for k, pts in enumerate(frames):
        alerts += det.update(pts if len(pts) else np.zeros((0, 5)), k * dt, sensor_xy=RADAR)
    return alerts


def test_walk_stand_walk_keeps_one_id_with_static_hold():
    rng = np.random.default_rng(3)
    frames = [person(rng, 1.0 + 0.04 * k, 0.0, 0.4) for k in range(40)]   # walks to 2.6 m
    # stands still 15 s: no points
    frames += [np.zeros((0, 5))] * 150
    frames += [person(rng, 2.6 + 0.04 * k, 0.2, 0.4) for k in range(40)]   # walks on
    det = Detector(open_room(), **IMPROVED)
    alerts = run(det, frames)
    assert len(alerts) == 1
    assert [tr.id for tr in det.confirmed_tracks()] == ['P1']


def test_static_track_is_held_in_place():
    rng = np.random.default_rng(4)
    frames = [person(rng, 2.0 + 0.03 * k, 1.0, 0.3) for k in range(30)] + [np.zeros((0, 5))] * 50
    det = Detector(open_room(), **IMPROVED)
    run(det, frames)
    (tr,) = det.confirmed_tracks()
    assert tr.life == 'static'
    assert np.linalg.norm(tr.pos - [2.9, 1.0]) < 0.5


def test_without_static_hold_the_stop_gives_a_second_alert():
    rng = np.random.default_rng(3)
    frames = [person(rng, 1.0 + 0.04 * k, 0.0, 0.4) for k in range(40)]
    # 20 s: longer than reacquire_s
    frames += [np.zeros((0, 5))] * 200
    frames += [person(rng, 2.6 + 0.04 * k, 0.2, 0.4) for k in range(40)]
    alerts = run(Detector(open_room(), **dict(IMPROVED, static_hold_s=0.0)), frames)
    assert len(alerts) == 2


def test_two_people_crossing_keep_their_ids():
    """
    A walks right to left at x=3.0, B left to right at x=3.5.

    They pass 0.5 m apart, moving in opposite directions (radial velocities -0.6 and +0.6 m/s).
    Each must keep its ID and not be merged into one.
    """
    rng = np.random.default_rng(5)
    det = Detector(open_room(), **IMPROVED)
    start_y = {}
    for k in range(80):
        a = person(rng, 3.0, -2.0 + 0.05 * k, -0.6)
        b = person(rng, 3.5, 2.0 - 0.05 * k, 0.6)
        for tr in det.update(np.vstack([a, b]), k * 0.1, sensor_xy=RADAR):
            start_y[tr.id] = tr.pos[1]
        if 35 <= k <= 45:   # while passing each other: still two people
            assert len(det.confirmed_tracks()) == 2
    assert len(start_y) == 2
    end_y = {tr.id: tr.pos[1] for tr in det.confirmed_tracks()}
    for tid, y0 in start_y.items():    # whoever started on one side ended on the other: no ID swap
        assert np.sign(end_y[tid]) == -np.sign(y0)


def test_far_and_weak_points_start_nothing():
    rng = np.random.default_rng(6)
    far = [person(rng, 8.0, 0.0, 0.5, n=6) for _ in range(40)]             # beyond max_range
    # 8 points over 4 frames < 10
    weak = [person(rng, 3.0, 2.5, 0.5, n=2) for _ in range(40)]
    assert run(Detector(open_room(), **IMPROVED), far) == []
    assert run(Detector(open_room(), **IMPROVED), weak) == []


def test_defaults_ignore_the_new_columns():
    """Room options off: (N, 5) input gives the same alerts as (N, 3)."""
    rng = np.random.default_rng(7)
    frames = [person(rng, 2.0 + 0.05 * k, 0.0, 0.3, n=6) for k in range(40)]
    a5 = run(Detector(open_room(), **ROOM), frames)
    a3 = run(Detector(open_room(), **ROOM), [f[:, :3] for f in frames])

    def ids_and_positions(alerts):
        return [(tr.id, tuple(np.round(tr.pos, 6))) for tr in alerts]

    assert ids_and_positions(a5) == ids_and_positions(a3)


def test_merge_close_and_assign():
    merged = merge_close([(np.array([0.0, 0.0]), 4), (np.array([0.5, 0.0]), 4),
                          (np.array([5.0, 0.0]), 3)], 1.0)
    assert len(merged) == 2 and np.allclose(merged[0][0], [0.25, 0.0]) and merged[0][1] == 8
    big = 1e9
    cost = np.array([[1.0, 2.0], [1.5, big]])
    # optimal (3.5), not greedy-first (row 0 takes col 0)
    assert sorted(assign(cost, big)) == [(0, 1), (1, 0)]


def test_max_range_boundary_is_inclusive():
    det = Detector(open_room(), **IMPROVED)   # max_range = 6.5 m
    pts = np.array([[6.5, 0.0, 1.0, 22.0, 0.4],     # exactly max_range
                    [6.6, 0.0, 1.0, 22.0, 0.4]])    # just beyond it
    kept = det.filter(pts, sensor_xy=RADAR)
    assert kept[:, 0].tolist() == [6.5]


@pytest.mark.parametrize('n_points, tracks_started', [(9, 0), (10, 1)])
def test_new_track_min_points_boundary(n_points, tracks_started):
    rng = np.random.default_rng(8)
    det = Detector(open_room(), **IMPROVED)   # new_track_min_points = 10
    det.update(person(rng, 3.0, 0.0, 0.4, n=n_points), 0.0, sensor_xy=RADAR)
    assert len(det.tracks) == tracks_started


@pytest.mark.parametrize('t, still_tracked', [(4.0, 1), (4.1, 0)])
def test_lost_after_boundary(t, still_tracked):
    det = Detector(open_room(), **ROOM)       # lost_after = 4.0 s
    track = Track('P1', (3.0, 0.0), 0.0)
    track.confirmed = True
    det.tracks = [track]
    det.update(np.zeros((0, 5)), t, sensor_xy=RADAR)
    assert len(det.tracks) == still_tracked
    assert len(det.lost) == 1 - still_tracked   # a confirmed track that expires is remembered


@pytest.mark.parametrize('t, remembered', [(14.9, 1), (15.0, 0)])
def test_lost_track_is_forgotten_after_reacquire_s(t, remembered):
    det = Detector(open_room(), **ROOM)       # reacquire_s = 15 s
    det.lost = [(Track('P1', (3.0, 0.0), 0.0), 0.0)]
    det.update(np.zeros((0, 5)), t, sensor_xy=RADAR)
    assert len(det.lost) == remembered


@pytest.mark.parametrize('frame', [
    np.full((6, 5), np.nan),         # every value missing
    np.array([1.0, 2.0, 3.0]),       # not a point array
    np.zeros((0, 5)),                # empty frame
], ids=['nan', 'not_2d', 'empty'])
def test_update_ignores_nan_and_malformed_frames(frame):
    det = Detector(open_room(), **IMPROVED)
    assert det.update(frame, 0.0, sensor_xy=RADAR) == []
    assert det.tracks == []
