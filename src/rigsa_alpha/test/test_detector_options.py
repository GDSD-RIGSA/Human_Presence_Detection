"""Detector branches the other detection tests do not reach (no ROS needed)."""
import sys

import numpy as np
import pytest

from rigsa_alpha.detection import (COAST_S, MAX_SPEED, MIN_POINTS, PREDICT_S, BackgroundMask,
                                   Detector, Track, assign)

ROOM = dict(gate=2.0, lost_after=4.0, reacquire_s=15.0, reacquire_m=3.0, eps=0.6)


def open_room():
    return BackgroundMask(np.zeros((320, 280), dtype=np.int8), 0.05, (-2.0, -8.0))


def blob(rng, x, y, n, vr=0.0, db=22.0, std=0.04):
    """n tight radar points around (x, y) as an (N, 5) map-frame array."""
    return np.column_stack([rng.normal(x, std, n), rng.normal(y, std, n), np.full(n, 1.0),
                            np.full(n, db), np.full(n, vr)])


def walking_frames(rng, n=20):
    return [blob(rng, 1.0 + 0.04 * k, 0.0, 4) for k in range(n)]


def feed(det, frames, t0=0.0, dt=0.1):
    alerts = []
    for k, pts in enumerate(frames):
        alerts += det.update(pts, t0 + k * dt, sensor_xy=(0.0, 0.0))
    return alerts


# ------------------------------------------------------------------------- set_state

def test_set_state_changes_only_the_track_with_that_event():
    # Arrange
    det = Detector(open_room(), **ROOM)
    rng = np.random.default_rng(1)
    first = feed(det, walking_frames(rng))[0]
    other = Track('P9', (5.0, 5.0), 0.0)
    other.event = first.event + 1
    det.tracks.append(other)
    # Act
    det.set_state(first.event, 'ack')
    # Assert
    assert first.state == 'ack'
    assert other.state == 'new'


def test_set_state_also_reaches_a_recently_lost_track():
    # Arrange: person walks, then nothing for longer than lost_after, so the track is "lost"
    det = Detector(open_room(), **ROOM)
    rng = np.random.default_rng(1)
    first = feed(det, walking_frames(rng))[0]
    feed(det, [np.zeros((0, 5))] * 50, t0=2.0)
    assert det.tracks == [] and [tr for tr, _ in det.lost] == [first]
    # Act
    det.set_state(first.event, 'false_alarm')
    # Assert
    assert first.state == 'false_alarm'


def test_set_state_unknown_event_changes_nothing():
    # Arrange
    det = Detector(open_room(), **ROOM)
    rng = np.random.default_rng(1)
    first = feed(det, walking_frames(rng))[0]
    # Act
    det.set_state(999, 'ack')
    # Assert
    assert first.state == 'new'


# ------------------------------------------------------------------------ update input

def test_update_accepts_xyz_only_points():
    # Arrange: (N, 3) frames, no intensity or velocity columns
    rng = np.random.default_rng(1)
    frames = [blob(rng, 1.0 + 0.04 * k, 0.0, 4)[:, :3] for k in range(20)]
    det = Detector(open_room(), **ROOM)
    # Act
    alerts = feed(det, frames)
    # Assert
    assert len(alerts) == 1


# --------------------------------------------------------------------------- cluster

def test_cluster_doppler_separates_people_moving_in_opposite_directions():
    # Arrange: two people at the same spot, one walking towards and one away from the radar
    rng = np.random.default_rng(2)
    acc = np.vstack([blob(rng, 2.0, 0.0, 8, vr=1.0), blob(rng, 2.0, 0.3, 8, vr=-1.0)])
    # Act
    together = Detector(open_room(), eps=0.6).cluster(acc)
    apart = Detector(open_room(), eps=0.6, doppler_k=1.0).cluster(acc)
    # Assert
    assert len(together) == 1
    assert len(apart) == 2


@pytest.mark.parametrize('n, kept', [(8, 0), (9, 1)])
def test_cluster_far_clusters_need_more_points(n, kept):
    # Arrange: at 6 m from the radar a cluster needs MIN_POINTS + 1.0 * (6 - 2) = 9 points
    assert MIN_POINTS == 5
    rng = np.random.default_rng(3)
    det = Detector(open_room(), min_points_per_m=1.0, min_points_r0=2.0)
    # Act
    clusters = det.cluster(blob(rng, 6.0, 0.0, n, std=0.0), sensor_xy=(0.0, 0.0))
    # Assert: the limit is strict, exactly 9 points still counts
    assert len(clusters) == kept


def test_cluster_near_clusters_keep_the_basic_point_limit():
    # Arrange: at 1 m the extra requirement is negative, 6 points are enough
    rng = np.random.default_rng(3)
    det = Detector(open_room(), min_points_per_m=1.0, min_points_r0=2.0)
    # Act
    clusters = det.cluster(blob(rng, 1.0, 0.0, 6, std=0.0), sensor_xy=(0.0, 0.0))
    # Assert
    assert len(clusters) == 1


def test_cluster_absorb_merges_two_blobs_of_one_person():
    # Arrange: two blobs 0.8 m apart, further than eps so DBSCAN keeps them apart
    rng = np.random.default_rng(4)
    acc = np.vstack([blob(rng, 2.0, 0.0, 8), blob(rng, 2.8, 0.0, 8)])
    # Act
    split = Detector(open_room(), eps=0.5).cluster(acc)
    merged = Detector(open_room(), eps=0.5, absorb_m=1.2).cluster(acc)
    # Assert: the merged blob has all points, at their centre
    assert len(split) == 2
    assert len(merged) == 1
    pos, n = merged[0]
    assert n == 16
    assert pos == pytest.approx([2.4, 0.0], abs=0.1)


# --------------------------------------------------------------------------- tracking

def test_greedy_distance_is_scaled_for_a_static_track():
    # Arrange: static tracks accept points up to static_radius, gate scaled to match
    det = Detector(open_room(), gate=1.2, static_radius=1.5)
    moving, static = Track('a', (0.0, 0.0), 0.0), Track('b', (0.0, 0.0), 0.0)
    static.life = 'static'
    # Act
    d_moving = det.distance(moving, np.array([1.5, 0.0]), 0.0)
    d_static = det.distance(static, np.array([1.5, 0.0]), 0.0)
    # Assert: exactly at the edge of static_radius gives exactly the gate
    assert d_moving == pytest.approx(1.5)
    assert d_static == pytest.approx(1.2)


def test_greedy_association_matches_the_closer_track_first():
    # Arrange: two tracks, one cluster between them but closer to the second
    det = Detector(open_room(), gate=1.2)
    a, b = Track('a', (0.0, 0.0), 0.0), Track('b', (1.0, 0.0), 0.0)
    det.tracks = [a, b]
    # Act
    used_t, used_c = det.associate_greedy([(np.array([0.7, 0.0]), 6)], 0.1)
    # Assert
    assert (used_t, used_c) == ({1}, {0})
    assert b.n_points == 6
    assert a.n_points == 0


def test_kf_predict_moves_a_track_by_its_velocity():
    # Arrange
    det = Detector(open_room(), tracker='kalman')
    tr = Track('a', (0.0, 0.0), 0.0)
    tr.kf_P = np.eye(4)
    tr.kf_x = np.array([0.0, 0.0, 1.0, 0.0])
    # Act: exactly PREDICT_S since the last update, still predicted
    det.kf_predict(tr, PREDICT_S)
    # Assert
    assert tr.kf_x[:2] == pytest.approx([1.0, 0.0])
    assert tr.kf_t == PREDICT_S


def test_kf_predict_stops_a_track_unseen_for_longer_than_predict_s():
    # Arrange
    det = Detector(open_room(), tracker='kalman')
    tr = Track('a', (0.0, 0.0), 0.0)
    tr.kf_P = np.eye(4)
    tr.kf_x = np.array([0.0, 0.0, 1.0, 0.0])
    # Act: just past the limit, the velocity is dropped before predicting
    det.kf_predict(tr, PREDICT_S + 0.01)
    # Assert
    assert tr.kf_x[:2] == pytest.approx([0.0, 0.0])


def test_kf_predict_does_not_move_a_static_track():
    # Arrange
    det = Detector(open_room(), tracker='kalman')
    tr = Track('a', (2.0, 1.0), 0.0)
    tr.life = 'static'
    tr.kf_P = np.eye(4)
    tr.kf_x = np.array([2.0, 1.0, 1.0, 1.0])
    # Act
    det.kf_predict(tr, 0.5)
    # Assert
    assert tr.kf_x[:2] == pytest.approx([2.0, 1.0])


def test_kf_predict_ignores_a_time_that_is_not_after_the_last_update():
    # Arrange
    det = Detector(open_room(), tracker='kalman')
    tr = Track('a', (0.0, 0.0), 1.0)
    tr.kf_P = np.eye(4)
    tr.kf_x = np.array([0.0, 0.0, 1.0, 0.0])
    # Act: dt == 0 is the boundary
    det.kf_predict(tr, 1.0)
    # Assert: nothing changed
    assert tr.kf_x.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert tr.kf_P.tolist() == np.eye(4).tolist()


def test_kf_predict_caps_the_speed_estimate():
    # Arrange
    det = Detector(open_room(), tracker='kalman')
    tr = Track('a', (0.0, 0.0), 0.0)
    tr.kf_P = np.eye(4)
    tr.kf_x = np.array([0.0, 0.0, 5.0, 0.0])
    # Act
    det.kf_predict(tr, 0.1)
    # Assert
    assert np.linalg.norm(tr.vel) == pytest.approx(MAX_SPEED)


def test_associate_kalman_starts_the_filter_of_a_track_that_has_none():
    # Arrange: a track created without Kalman state, e.g. by the greedy tracker
    det = Detector(open_room(), tracker='kalman')
    tr = Track('a', (0.0, 0.0), 0.0)
    det.tracks = [tr]
    # Act
    used_t, used_c = det.associate_kalman([(np.array([0.1, 0.0]), 7)], 0.1)
    # Assert
    assert tr.kf_P is not None
    assert (used_t, used_c) == ({0}, {0})
    assert tr.n_points == 7


def test_associate_kalman_leaves_a_cluster_outside_the_gate_unmatched():
    # Arrange
    det = Detector(open_room(), tracker='kalman')
    det.tracks = [Track('a', (0.0, 0.0), 0.0)]
    # Act
    used_t, used_c = det.associate_kalman([(np.array([8.0, 8.0]), 7)], 0.1)
    # Assert
    assert (used_t, used_c) == (set(), set())


# ----------------------------------------------------------------------------- assign

def test_assign_hungarian_minimises_the_total_cost():
    # Arrange: greedy would take the 1 first; the cheapest total is 5 + 2
    cost = np.array([[1.0, 5.0], [2.0, 1e9]])
    # Act
    pairs = assign(cost, 1e9)
    # Assert
    assert sorted((int(r), int(c)) for r, c in pairs) == [(0, 1), (1, 0)]


def test_assign_without_scipy_falls_back_to_greedy(monkeypatch):
    # Arrange: make "from scipy.optimize import ..." raise ImportError
    monkeypatch.setitem(sys.modules, 'scipy.optimize', None)
    cost = np.array([[1.0, 5.0], [2.0, 1e9]])
    # Act
    pairs = assign(cost, 1e9)
    # Assert: cheapest pair first, then nothing else fits
    assert [(int(r), int(c)) for r, c in pairs] == [(0, 0)]


def test_assign_without_scipy_matches_each_row_and_column_once(monkeypatch):
    # Arrange
    monkeypatch.setitem(sys.modules, 'scipy.optimize', None)
    cost = np.array([[1.0, 2.0], [1.5, 3.0]])
    # Act
    pairs = assign(cost, 1e9)
    # Assert
    assert [(int(r), int(c)) for r, c in pairs] == [(0, 0), (1, 1)]


# --------------------------------------------------------------------------- reacquire

def confirmed_track(tid, pos, last):
    tr = Track(tid, pos, last)
    tr.confirmed = True
    return tr


def test_reacquire_takes_over_a_coasting_track_and_removes_it():
    # Arrange: old track not updated for more than COAST_S, new one 1 m away
    det = Detector(open_room(), **ROOM)
    old, new = confirmed_track('P1', (0.0, 0.0), 0.0), Track('x', (1.0, 0.0), 1.0)
    det.tracks = [old, new]
    # Act
    found = det.reacquire(new, 1.0)
    # Assert
    assert found is old
    assert det.tracks == [new]


def test_reacquire_ignores_a_track_updated_within_coast_s():
    # Arrange: boundary, exactly COAST_S since the last update is not coasting yet
    det = Detector(open_room(), **ROOM)
    old, new = confirmed_track('P1', (0.0, 0.0), 1.0 - COAST_S), Track('x', (1.0, 0.0), 1.0)
    det.tracks = [old, new]
    # Act
    found = det.reacquire(new, 1.0)
    # Assert
    assert found is None
    assert det.tracks == [old, new]


@pytest.mark.parametrize('dist, taken', [(2.99, True), (3.0, False)])
def test_reacquire_distance_limit_is_strict(dist, taken):
    # Arrange: reacquire_m is 3.0 in ROOM
    det = Detector(open_room(), **ROOM)
    old, new = confirmed_track('P1', (0.0, 0.0), 0.0), Track('x', (dist, 0.0), 1.0)
    det.tracks = [old, new]
    # Act
    found = det.reacquire(new, 1.0)
    # Assert
    assert (found is old) is taken


def test_reacquire_prefers_the_nearest_candidate():
    # Arrange: a coasting track 2 m away and a lost track 1 m away
    det = Detector(open_room(), **ROOM)
    coasting = confirmed_track('P1', (2.0, 0.0), 0.0)
    lost = confirmed_track('P2', (-1.0, 0.0), 0.0)
    new = Track('x', (0.0, 0.0), 1.0)
    det.tracks = [coasting, new]
    det.lost = [(lost, 0.5)]
    # Act
    found = det.reacquire(new, 1.0)
    # Assert: the lost track wins and leaves the lost list
    assert found is lost
    assert det.lost == []
    assert coasting in det.tracks


# ----------------------------------------------------------------- update: new tracks

@pytest.mark.parametrize('absorb_m, tracks', [(1.2, 1), (0.0, 2)])
def test_update_blob_near_a_confirmed_track_does_not_start_a_second_person(absorb_m, tracks):
    # Arrange: a confirmed person at (2, 0); a new blob 1 m away is outside the small gate
    rng = np.random.default_rng(5)
    det = Detector(open_room(), gate=0.3, absorb_m=absorb_m)
    known = Track('P1', (2.0, 0.0), 0.0)
    known.confirmed = True
    det.tracks = [known]
    # Act
    det.update(blob(rng, 3.0, 0.0, 8), 0.1, sensor_xy=(0.0, 0.0))
    # Assert: absorbed when absorb_m covers the 1 m, otherwise tracked as a new candidate
    assert len(det.tracks) == tracks


def test_update_same_person_after_a_short_gap_keeps_id_and_raises_no_new_alert():
    # Arrange: walks, disappears for 4.5 s (longer than lost_after, shorter than reacquire_s),
    # then reappears close to where they vanished
    rng = np.random.default_rng(1)
    frames = walking_frames(rng)
    frames += [np.zeros((0, 5))] * 45
    frames += [blob(rng, 1.8 + 0.04 * k, 0.0, 4) for k in range(20)]
    det = Detector(open_room(), **ROOM)
    # Act
    alerts = feed(det, frames)
    # Assert
    assert len(alerts) == 1
    (tr,) = det.confirmed_tracks()
    assert (tr.id, tr.event) == (alerts[0].id, alerts[0].event) == ('P1', 1)
