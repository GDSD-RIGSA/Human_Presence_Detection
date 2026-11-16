"""The scripted alpha scenario: polyline walking, B2 patrol pose and people poses (no ROS)."""
import math

import pytest

from rigsa_alpha.scenario import (PATROL, PATROL_SPEED, PEOPLE, SCENARIO_S, TURN_S, WALK, along,
                                  people_poses, robot_pose)

L_PATH = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]   # 4 m east, then 3 m north
WALK_S = math.hypot(PATROL[1][0] - PATROL[0][0], PATROL[1][1] - PATROL[0][1]) / PATROL_SPEED


def path_length(path):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def test_along_start_of_the_path():
    # Arrange / Act
    x, y, heading = along(L_PATH, 0.0)
    # Assert
    assert (x, y, heading) == (0.0, 0.0, 0.0)


def test_along_middle_of_a_segment():
    # Arrange / Act
    x, y, heading = along(L_PATH, 1.0)
    # Assert
    assert (x, y) == pytest.approx((1.0, 0.0))
    assert heading == pytest.approx(0.0)


def test_along_exactly_at_a_corner_belongs_to_the_first_segment():
    # Arrange / Act: boundary, dist == segment length
    x, y, heading = along(L_PATH, 4.0)
    # Assert: still heading east, not yet north
    assert (x, y) == pytest.approx((4.0, 0.0))
    assert heading == pytest.approx(0.0)


def test_along_after_the_corner_turns_to_the_next_segment():
    # Arrange / Act
    x, y, heading = along(L_PATH, 5.5)
    # Assert
    assert (x, y) == pytest.approx((4.0, 1.5))
    assert heading == pytest.approx(math.pi / 2)


def test_along_exactly_at_the_end_is_still_on_the_path():
    # Arrange / Act
    pose = along(L_PATH, 7.0)
    # Assert
    assert pose[:2] == pytest.approx((4.0, 3.0))


@pytest.mark.parametrize('dist', [-0.001, 7.001])
def test_along_outside_the_path_is_none(dist):
    # Arrange / Act / Assert: before the start and just past the end
    assert along(L_PATH, dist) is None


def test_along_path_with_a_single_point_is_none():
    # Arrange / Act / Assert: no segment to walk on
    assert along([(1.0, 1.0)], 0.0) is None


def test_robot_pose_starts_at_the_first_patrol_point():
    # Arrange / Act
    x, y, yaw = robot_pose(0.0)
    # Assert
    assert (x, y) == pytest.approx(PATROL[0])
    assert yaw == pytest.approx(0.0)


def test_robot_pose_halfway_along_the_hallway():
    # Arrange / Act
    x, y, yaw = robot_pose(WALK_S / 2)
    # Assert
    assert (x, y) == pytest.approx((0.0, PATROL[0][1]))
    assert yaw == pytest.approx(0.0)


def test_robot_pose_at_the_end_of_the_walk_starts_turning_on_the_spot():
    # Arrange / Act: boundary between walking and turning
    x, y, yaw = robot_pose(WALK_S)
    # Assert
    assert (x, y) == pytest.approx(PATROL[1])
    assert yaw == pytest.approx(0.0)


def test_robot_pose_turn_rotates_half_a_circle_in_turn_s():
    # Arrange / Act
    mid = robot_pose(WALK_S + TURN_S / 2)
    done = robot_pose(WALK_S + TURN_S)
    # Assert: stays at the end point; the B2 faces back along the hallway after the turn
    assert mid[:2] == pytest.approx(PATROL[1])
    assert mid[2] == pytest.approx(math.pi / 2)
    assert done[:2] == pytest.approx(PATROL[1])
    assert done[2] == pytest.approx(math.pi)


def test_robot_pose_walks_back_and_repeats_after_a_full_cycle():
    # Arrange
    period = 2 * (WALK_S + TURN_S)
    # Act
    back = robot_pose(WALK_S + TURN_S + WALK_S / 2)
    again = robot_pose(7.0 + period)
    # Assert
    assert back[:2] == pytest.approx((0.0, PATROL[0][1]))
    assert back[2] == pytest.approx(math.pi)
    assert again == pytest.approx(robot_pose(7.0))


def test_people_poses_before_anyone_starts_is_all_none():
    # Arrange / Act
    poses = people_poses(0.0)
    # Assert
    assert poses == [None] * len(PEOPLE)


def test_people_poses_person_appears_at_the_start_of_their_path():
    # Arrange
    start = PEOPLE[0]['start']
    # Act
    poses = people_poses(start)
    # Assert: only the first person is out, standing on the first path point
    assert poses[0][:2] == pytest.approx(PEOPLE[0]['path'][0])
    assert poses[1] is None and poses[2] is None


def test_people_poses_person_leaves_after_their_path_ends():
    # Arrange: first person walks their whole path, then is gone
    end = PEOPLE[0]['start'] + path_length(PEOPLE[0]['path']) / WALK
    # Act
    at_end = people_poses(end - 1e-6)
    after = people_poses(end + 0.1)
    # Assert
    assert at_end[0][:2] == pytest.approx(PEOPLE[0]['path'][-1])
    assert after[0] is None


def test_people_poses_scene_repeats_every_scenario_s():
    # Arrange / Act / Assert
    assert people_poses(30.0 + SCENARIO_S) == people_poses(30.0)
