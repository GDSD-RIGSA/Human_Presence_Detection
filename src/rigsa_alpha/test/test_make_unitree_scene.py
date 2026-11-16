"""`make_unitree_scene`: the generated unitree_mujoco scene (pure XML building, no simulator)."""
import sys
import xml.etree.ElementTree as ET

import pytest

pytest.importorskip('mujoco')

from rigsa_alpha import make_unitree_scene as scene  # noqa: E402
from rigsa_alpha.floor import FURNITURE, WALLS  # noqa: E402
from rigsa_alpha.mujoco_world import building_geoms  # noqa: E402
from rigsa_alpha.scenario import SENTRY_POST  # noqa: E402


def test_scene_includes_the_b2_and_every_wall_and_furniture():
    # Arrange
    xml = scene.scene_xml()
    geoms = building_geoms(WALLS, FURNITURE, dx=-SENTRY_POST[0], dy=-SENTRY_POST[1])
    # Act / Assert
    assert '<include file="b2.xml"/>' in xml
    assert len(geoms) == len(WALLS) + len(FURNITURE)
    for geom in geoms:
        assert geom in xml
    assert ET.fromstring(xml).tag == 'mujoco'          # well-formed, include left unresolved


def test_building_is_shifted_so_the_sim_origin_is_the_sentry_post():
    # Arrange: the B2 starts at the simulator origin, so the building moves by -SENTRY_POST
    xml = scene.scene_xml()
    first_wall = building_geoms(WALLS, FURNITURE, dx=-SENTRY_POST[0], dy=-SENTRY_POST[1])[0]
    unshifted = building_geoms(WALLS, FURNITURE)[0]
    x1, y1, x2, y2 = WALLS[0]
    # Act / Assert: the wall centre is the old centre minus the sentry post
    assert first_wall != unshifted
    assert 'map x=%s, y=%s' % (SENTRY_POST[0], SENTRY_POST[1]) in xml
    assert 'pos="%.3f %.3f' % ((x1 + x2) / 2 - SENTRY_POST[0],
                               (y1 + y2) / 2 - SENTRY_POST[1]) in first_wall


def test_a_rotated_sentry_post_is_refused(monkeypatch):
    # Arrange: the template only shifts, it cannot rotate the building
    monkeypatch.setattr(scene, 'SENTRY_POST', (-2.0, 0.0, 0.5))
    # Act / Assert
    with pytest.raises(SystemExit) as err:
        scene.scene_xml()
    assert 'yaw at 0' in str(err.value.code)


def test_main_writes_the_file_and_prints_the_path(tmp_path, monkeypatch, capsys):
    # Arrange
    target = tmp_path / 'scene_rigsa.xml'
    monkeypatch.setattr(sys, 'argv', ['make_unitree_scene', str(target)])
    # Act
    scene.main()
    # Assert
    assert target.read_text() == scene.scene_xml()
    assert capsys.readouterr().out.strip() == 'wrote %s' % target


def test_main_prints_the_xml_without_a_path(monkeypatch, capsys):
    # Arrange
    monkeypatch.setattr(sys, 'argv', ['make_unitree_scene'])
    # Act
    scene.main()
    # Assert
    assert capsys.readouterr().out.strip() == scene.scene_xml().strip()


def test_an_option_is_not_taken_as_an_output_path(tmp_path, monkeypatch, capsys):
    # Arrange: an argument starting with -- must not become a file name
    monkeypatch.setattr(sys, 'argv', ['make_unitree_scene', '--help'])
    # Act
    scene.main()
    # Assert
    assert '<mujoco' in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []
