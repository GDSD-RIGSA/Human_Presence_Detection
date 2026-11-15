"""MuJoCo radar world: scene building, robot / people placement, radar scan, line of sight.

Runs headless (no viewer, no display). Skipped where the `mujoco` package is not installed.
"""
import math

import numpy as np
import pytest

pytest.importorskip('mujoco')
import mujoco  # noqa: E402
from rigsa_alpha.floor import FURNITURE, WALLS  # noqa: E402
from rigsa_alpha.mujoco_world import (JOINTS, RADAR_RANGE, World, build_mjcf,  # noqa: E402
                                      building_geoms, multi_ray, people_bodies, simple_b2)

EYE = np.eye(3)
FACING_AWAY = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])   # radar looks along -x
ORIGIN = (0.0, 0.0, 1.0)
STRONG = 12.0    # dB; people reflect at 16 dB, walls 11, floor 6, ghosts under 9


def make_world(walls=(), furniture=(), n_people=1, seed=0):
    world = World(list(walls), list(furniture), n_people, seed=seed)
    world.forward()   # geom positions are only computed by a forward pass
    return world


def scan_many(world, rot=EYE, frames=30, origin=ORIGIN, sensor_vel=(0, 0, 0)):
    """Several radar frames stacked, as (N, 5) in the radar frame."""
    return np.vstack([world.radar_scan(origin, rot, sensor_vel) for _ in range(frames)])


def person_world(x, y, walls=(), vel=(0.0, 0.0, 0.0)):
    w = make_world(walls)
    w.set_people([(x, y, 0.0)], [vel])
    w.forward()
    return w


def near(pts, x, y, r=0.6):
    return pts[(np.hypot(pts[:, 0] - x, pts[:, 1] - y) < r) & (pts[:, 3] > STRONG)]


# ---------------------------------------------------------------------------- MJCF

def test_building_geoms_one_box_per_wall_and_piece_of_furniture():
    # Arrange / Act
    geoms = building_geoms([(0, 0, 4, 0), (4, 0, 4, 3)], [(1, 1, 2, 1)])
    # Assert
    assert len(geoms) == 3
    assert 'name="wall0"' in geoms[0] and 'name="wall1"' in geoms[1]
    assert 'name="furniture0"' in geoms[2]


def test_building_geoms_wall_length_position_and_heading():
    # Arrange / Act: a 4 m wall along y at x = 4
    (geom,) = building_geoms([(4, 0, 4, 4)], [])
    # Assert: centre (4, 2), half length 2 + 0.05 end cap, turned 90 degrees
    assert 'pos="4.000 2.000' in geom
    assert 'size="2.050 0.05 ' in geom
    assert 'euler="0 0 90.00"' in geom


def test_building_geoms_shift_moves_everything():
    # Arrange / Act
    (geom,) = building_geoms([(0, 0, 2, 0)], [], dx=10.0, dy=-5.0)
    # Assert
    assert 'pos="11.000 -5.000' in geom


def test_people_bodies_one_mocap_body_per_person_and_none_for_zero():
    # Arrange / Act / Assert
    xml = people_bodies(3)
    assert xml.count('mocap="true"') == 3
    assert 'name="person2"' in xml
    assert people_bodies(0) == ''


def test_simple_b2_has_four_legs_and_the_twelve_unitree_joints():
    # Arrange / Act
    xml = simple_b2()
    # Assert
    assert 'name="base_link"' in xml and 'freejoint' in xml
    for joint in JOINTS:
        assert 'name="%s"' % joint in xml


def test_build_mjcf_without_b2_file_uses_the_simple_b2_and_compiles():
    # Arrange / Act
    xml = build_mjcf(WALLS, FURNITURE, 3, b2_xml='/no/such/b2.xml')
    model = mujoco.MjModel.from_xml_string(xml)
    # Assert
    assert 'RIGSA radar world' in xml
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link') >= 0


def test_build_mjcf_merges_the_world_into_a_given_b2_file(tmp_path):
    # Arrange: a minimal stand-in for unitree's b2.xml
    b2 = tmp_path / 'b2.xml'
    b2.write_text('<mujoco model="b2"><compiler meshdir="assets_old"/>'
                  '<worldbody><body name="base_link"><freejoint/>'
                  '<geom type="box" size="0.1 0.1 0.1"/></body></worldbody></mujoco>')
    # Act
    xml = build_mjcf([(0, 0, 2, 0)], [], 1, b2_xml=str(b2))
    # Assert: model renamed, mesh folder made absolute, our floor and walls added to its world
    assert 'model="RIGSA radar world"' in xml
    assert 'meshdir="%s"' % (tmp_path / 'assets') in xml
    assert 'name="floor"' in xml and 'name="wall0"' in xml and 'name="person0"' in xml


def test_multi_ray_falls_back_to_the_older_signature(monkeypatch):
    # Arrange: MuJoCo before 3.3 has no `normal` argument; the new call raises TypeError
    calls = []

    def fake(*args):
        calls.append(len(args))
        if len(args) == 12:
            raise TypeError('old mujoco')

    monkeypatch.setattr(mujoco, 'mj_multiRay', fake)
    # Act
    multi_ray(None, None, None, None, None, None, None, 1, 10.0)
    # Assert
    assert calls == [12, 11]


# ---------------------------------------------------------------------------- World

def test_world_builds_the_real_floor_plan():
    # Arrange / Act
    world = World(WALLS, FURNITURE, 3, seed=1)
    # Assert
    assert world.geom_kind.count('wall') == len(WALLS)
    assert world.geom_kind.count('furniture') == len(FURNITURE)
    assert world.geom_kind.count('floor') == 1
    assert len(world.person_mocap) == 3
    assert world.full_b2 is False


def test_world_classifies_person_geoms_and_robot_geoms():
    # Arrange / Act
    world = make_world(n_people=2)
    people = [i for i, k in enumerate(world.geom_kind) if k == 'person']
    # Assert: body and head per person, mapped back to the person's number
    assert sorted(set(world.geom_person[people])) == [0, 1]
    assert len(people) == 4
    assert 'robot' in world.geom_kind
    assert world.p_detect[people[0]] == pytest.approx(0.6)


def test_world_seed_makes_scans_repeatable():
    # Arrange
    a, b = person_world(3.0, 0.0), person_world(3.0, 0.0)
    # Act
    pa = a.radar_scan(ORIGIN, EYE, (0, 0, 0))
    pb = b.radar_scan(ORIGIN, EYE, (0, 0, 0))
    # Assert
    assert np.array_equal(pa, pb)


def test_set_robot_writes_the_base_pose_and_known_joints_only():
    # Arrange
    world = make_world()
    # Act
    world.set_robot((1.0, 2.0, 0.5), (1.0, 0.0, 0.0, 0.0), {'FR_hip_joint': 0.3, 'bogus': 9.0})
    # Assert
    q = world.data.qpos
    assert q[world.root_q:world.root_q + 7].tolist() == [1.0, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0]
    assert q[world.joint_q['FR_hip_joint']] == pytest.approx(0.3)


def test_set_people_places_heading_and_parks_absent_people_below_the_floor():
    # Arrange
    world = make_world(n_people=2)
    # Act
    world.set_people([(3.0, 1.0, math.pi / 2), None], np.zeros((2, 3)))
    # Assert
    a, b = world.person_mocap
    assert world.data.mocap_pos[a].tolist() == [3.0, 1.0, 0.0]
    assert world.data.mocap_quat[a] == pytest.approx([math.cos(math.pi / 4), 0, 0,
                                                      math.sin(math.pi / 4)])
    assert world.data.mocap_pos[b].tolist() == [0.0, 0.0, -10.0]


def test_set_people_velocities_are_reshaped_to_n_by_3():
    # Arrange
    world = make_world(n_people=2)
    # Act
    world.set_people([None, None], [[1, 0, 0], [0, 2, 0]])
    # Assert
    assert world.people_vel.shape == (2, 3)
    assert world.people_vel[1].tolist() == [0.0, 2.0, 0.0]


# ------------------------------------------------------------------------ radar_scan

def test_scan_returns_float32_points_with_five_columns():
    # Arrange
    world = person_world(3.0, 0.0)
    # Act
    pts = world.radar_scan(ORIGIN, EYE, (0, 0, 0))
    # Assert
    assert pts.dtype == np.float32
    assert pts.ndim == 2 and pts.shape[1] == 5


def test_scan_sees_a_person_in_front_at_about_the_right_place_and_snr():
    # Arrange
    world = person_world(3.0, 0.0)
    # Act
    pts = near(scan_many(world), 3.0, 0.0)
    # Assert
    assert len(pts) > 0
    assert pts[:, 3].mean() == pytest.approx(16.0, abs=2.0)
    assert np.hypot(pts[:, 0] - 3.0, pts[:, 1]).max() < 0.6


def test_scan_does_not_see_a_person_behind_the_radar():
    # Arrange: the radar looks along -x, the person stands at +x
    world = person_world(3.0, 0.0)
    # Act
    pts = scan_many(world, rot=FACING_AWAY)
    # Assert
    assert len(near(pts, 3.0, 0.0)) == 0
    assert (pts[:, 3] > STRONG).sum() == 0


def test_scan_does_not_see_a_person_beyond_the_range():
    # Arrange: boundary, RADAR_RANGE is 10 m
    world = person_world(RADAR_RANGE + 2.0, 0.0)
    # Act
    pts = scan_many(world)
    # Assert
    assert (pts[:, 3] > STRONG).sum() == 0


def test_scan_does_not_see_a_person_outside_the_field_of_view():
    # Arrange: 3 m to the left, 90 degrees off the +-60 degree beam
    world = person_world(0.0, 3.0)
    # Act
    pts = scan_many(world)
    # Assert
    assert (pts[:, 3] > STRONG).sum() == 0


def test_scan_wall_hides_the_person_behind_it():
    # Arrange: wall across the beam at x = 2, person at x = 4
    world = person_world(4.0, 0.0, walls=[(2, -4, 2, 4)])
    # Act
    pts = scan_many(world)
    # Assert: no strong return from the person's position beyond the wall
    assert len(near(pts, 4.0, 0.0)) == 0


def test_scan_wall_points_appear_at_the_wall_distance():
    # Arrange
    world = make_world(walls=[(5, -6, 5, 6)])
    # Act
    pts = scan_many(world, frames=60)
    wall = pts[(pts[:, 0] > 4.5) & (pts[:, 0] < 5.5) & (pts[:, 2] > 0.2) & (pts[:, 3] > 9.5)]
    # Assert: weak returns (about 11 dB, above the ghosts and the floor), but present
    assert len(wall) > 0
    assert 9.5 < wall[:, 3].mean() < 14.0


def test_scan_radial_velocity_of_a_person_walking_away_is_positive():
    # Arrange: moving away from the radar at 1 m/s
    world = person_world(3.0, 0.0, vel=(1.0, 0.0, 0.0))
    # Act
    pts = near(scan_many(world), 3.0, 0.0)
    # Assert
    assert len(pts) > 0
    assert pts[:, 4].mean() == pytest.approx(1.0, abs=0.15)


def test_scan_radial_velocity_includes_the_motion_of_the_radar():
    # Arrange: still person, radar driving towards them at 0.5 m/s
    world = person_world(3.0, 0.0)
    # Act
    pts = near(scan_many(world, sensor_vel=(0.5, 0, 0)), 3.0, 0.0)
    # Assert: approaching is negative
    assert len(pts) > 0
    assert pts[:, 4].mean() == pytest.approx(-0.5, abs=0.1)


# --------------------------------------------------------------------- line_of_sight

def test_line_of_sight_clear_view_is_true():
    # Arrange
    world = person_world(8.0, 5.0)
    # Act / Assert
    assert world.line_of_sight(ORIGIN, (4.0, 0.0, 1.0))


def test_line_of_sight_wall_in_between_is_false():
    # Arrange
    world = person_world(8.0, 5.0, walls=[(2, -4, 2, 4)])
    # Act / Assert
    assert not world.line_of_sight(ORIGIN, (4.0, 0.0, 1.0))


def test_line_of_sight_target_in_front_of_the_wall_is_true():
    # Arrange: boundary, the wall is further away than the target
    world = person_world(8.0, 5.0, walls=[(2, -4, 2, 4)])
    # Act / Assert
    assert world.line_of_sight(ORIGIN, (1.0, 0.0, 1.0))


def test_line_of_sight_to_a_person_counts_as_seen():
    # Arrange: the first thing hit on the way to the target is the person themselves
    world = person_world(3.0, 0.0)
    # Act / Assert
    assert world.line_of_sight(ORIGIN, (3.0, 0.0, 1.0))
