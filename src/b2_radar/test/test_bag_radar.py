import os

import pytest

from b2_radar.bag_radar_node import read_bag

BAGS = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'bags')


@pytest.mark.parametrize('name', ['s1_empty_room', 's2_one_walking', 's3_walk_stand_walk'])
def test_recorded_bags_replay_the_same_radar_frames(name):
    pytest.importorskip('rosbag2_py')
    path = os.path.join(BAGS, name)
    first, second = read_bag(path), read_bag(path)
    assert len(first) > 100                                   # ~10 Hz for 45 s
    assert [t for t, _ in first] == [t for t, _ in second]    # same frames, same order, every time
    assert all(b[0] >= a[0] for a, b in zip(first, first[1:]))
