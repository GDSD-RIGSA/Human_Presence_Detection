"""Operator panel: alert display, acknowledge buttons, footers and the event table.

Runs the real Qt widgets with the offscreen platform (no display) on a fake ROS node and a
temporary SQLite event store. Skipped where Qt (python_qt_binding) is not installed.
"""
import json
import os
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('rclpy')
pytest.importorskip('python_qt_binding.QtWidgets')
import rclpy  # noqa: E402
from python_qt_binding.QtWidgets import QApplication  # noqa: E402
from rigsa_alpha import operator_panel  # noqa: E402
from rigsa_alpha.event_store import EventStore  # noqa: E402
from rigsa_alpha.operator_panel import OperatorPanel  # noqa: E402

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope='session')
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def store(tmp_path):
    s = EventStore('host=127.0.0.1 port=1 connect_timeout=1', str(tmp_path / 'events.db'))
    assert s.backend == 'sqlite'
    yield s
    s.close()


@pytest.fixture
def panel(qapp, store, monkeypatch):
    monkeypatch.setattr(QApplication, 'beep', staticmethod(lambda: None))
    published = []
    node = SimpleNamespace(
        create_publisher=lambda *a, **k: SimpleNamespace(publish=published.append),
        create_subscription=lambda *a, **k: None)
    p = OperatorPanel(node, store, 'alice')
    p.published = published
    yield p
    p.close()


def msg(payload):
    return SimpleNamespace(data=payload if isinstance(payload, str) else json.dumps(payload))


def alert(event=1, track='P1', **kw):
    a = dict(event=event, track=track, distance=3.14, direction='20° left', time='12:00:00')
    a.update(kw)
    return a


def db_alert(event, track='P1', position=True):
    a = dict(event=event, track=track, stamp=T0.isoformat(), n_points=9, robot_x=0.0,
             robot_y=0.5, robot_yaw=0.1, source='radar')
    if position:
        a.update(x=2.5, y=-1.0, distance=3.1, bearing_deg=-20.0)
    return a


# --------------------------------------------------------------------------- start up

def test_panel_starts_with_no_alert_and_disabled_buttons(panel):
    # Arrange / Act: constructed by the fixture
    # Assert
    assert panel.alert_title.text() == 'No active alerts'
    assert panel.alert_detail.text() == 'Waiting for the radar…'
    assert panel.alert_time.text() == ''
    assert not panel.ack_btn.isEnabled()
    assert not panel.fa_btn.isEnabled()


def test_panel_starts_with_the_footers_showing_no_data(panel):
    # Arrange / Act / Assert
    assert 'No data from the robot' in panel.footer.text()
    assert 'Event log not running' in panel.db_footer.text()


# ----------------------------------------------------------------------------- on_alert

def test_on_alert_shows_the_new_alert_and_enables_the_buttons(panel):
    # Arrange / Act
    panel.on_alert(msg(alert(event=4, track='P3')))
    # Assert
    assert panel.current['status'] == 'new'
    assert 'Person detected' in panel.alert_title.text()
    assert panel.alert_detail.text() == 'P3 · 3.1 m · 20° left'
    assert panel.alert_time.text() == '12:00:00'
    assert panel.ack_btn.isEnabled() and panel.fa_btn.isEnabled()


def test_on_alert_without_a_position_says_so(panel):
    # Arrange / Act
    panel.on_alert(msg(alert(distance=None, direction=None)))
    # Assert
    assert panel.alert_detail.text() == 'P1 · position unknown'


def test_on_alert_with_a_distance_but_no_direction_says_position_unknown(panel):
    # Arrange / Act: both are needed to show the position
    panel.on_alert(msg(alert(direction='')))
    # Assert
    assert panel.alert_detail.text() == 'P1 · position unknown'


def test_on_alert_ignores_text_that_is_not_json(panel):
    # Arrange / Act
    panel.on_alert(msg('{broken'))
    # Assert
    assert panel.current is None


def test_on_alert_replaces_the_previous_alert(panel):
    # Arrange
    panel.on_alert(msg(alert(event=1)))
    panel.handle('ack')
    # Act
    panel.on_alert(msg(alert(event=2, track='P2')))
    # Assert
    assert panel.current['event'] == 2 and panel.current['status'] == 'new'
    assert panel.ack_btn.isEnabled()


# -------------------------------------------------------------------------------- handle

@pytest.mark.parametrize('action, title', [('ack', 'Acknowledged'),
                                           ('false_alarm', 'Marked as false alarm')])
def test_handle_publishes_the_operator_action_and_closes_the_alert(panel, action, title):
    # Arrange
    panel.on_alert(msg(alert(event=7)))
    # Act
    panel.handle(action)
    # Assert
    (sent,) = panel.published
    assert json.loads(sent.data) == dict(event=7, action=action, operator='alice')
    assert panel.current['status'] == action
    assert panel.alert_title.text() == title
    assert not panel.ack_btn.isEnabled() and not panel.fa_btn.isEnabled()


def test_handle_without_an_alert_does_nothing(panel):
    # Arrange / Act
    panel.handle('ack')
    # Assert
    assert panel.published == []


def test_handle_an_alert_that_was_already_handled_does_nothing(panel):
    # Arrange
    panel.on_alert(msg(alert()))
    panel.handle('ack')
    # Act
    panel.handle('false_alarm')
    # Assert: only the first action was sent
    assert len(panel.published) == 1
    assert panel.current['status'] == 'ack'


def test_clicking_the_buttons_sends_ack_and_false_alarm(panel):
    # Arrange
    panel.on_alert(msg(alert(event=1)))
    panel.ack_btn.click()
    panel.on_alert(msg(alert(event=2)))
    # Act
    panel.fa_btn.click()
    # Assert
    assert [json.loads(m.data)['action'] for m in panel.published] == ['ack', 'false_alarm']


# --------------------------------------------------------------------------------- on_ack

def test_on_ack_from_another_operator_closes_the_matching_alert(panel):
    # Arrange
    panel.on_alert(msg(alert(event=5)))
    # Act
    panel.on_ack(msg(dict(event=5, action='false_alarm', operator='bob')))
    # Assert
    assert panel.current['status'] == 'false_alarm'
    assert not panel.ack_btn.isEnabled()
    assert panel.published == []


def test_on_ack_without_an_action_defaults_to_ack(panel):
    # Arrange
    panel.on_alert(msg(alert(event=5)))
    # Act
    panel.on_ack(msg(dict(event=5)))
    # Assert
    assert panel.current['status'] == 'ack'


def test_on_ack_for_a_different_event_is_ignored(panel):
    # Arrange
    panel.on_alert(msg(alert(event=5)))
    # Act
    panel.on_ack(msg(dict(event=6, action='ack')))
    # Assert
    assert panel.current['status'] == 'new'


def test_on_ack_does_not_overwrite_an_alert_that_was_already_handled(panel):
    # Arrange
    panel.on_alert(msg(alert(event=5)))
    panel.handle('ack')
    # Act
    panel.on_ack(msg(dict(event=5, action='false_alarm')))
    # Assert
    assert panel.current['status'] == 'ack'


def test_on_ack_with_no_alert_or_bad_json_is_ignored(panel):
    # Arrange / Act
    panel.on_ack(msg(dict(event=1, action='ack')))
    panel.on_ack(msg('nope'))
    # Assert
    assert panel.current is None


# ------------------------------------------------------------------------ status callbacks

def test_on_status_stores_the_robot_status_and_the_time(panel):
    # Arrange
    before = time.monotonic()
    # Act
    panel.on_status(msg(dict(robot='B2 sim', height_m=0.5)))
    # Assert
    assert panel.status == dict(robot='B2 sim', height_m=0.5)
    assert panel.last_status >= before


def test_on_status_with_bad_json_keeps_the_old_time(panel):
    # Arrange
    panel.on_status(msg(dict(robot='B2')))
    first = panel.last_status
    # Act
    panel.on_status(msg('nope'))
    # Assert
    assert panel.last_status == first and panel.status == dict(robot='B2')


def test_on_log_status_stores_the_log_status(panel):
    # Arrange / Act
    panel.on_log_status(msg(dict(ok=True, run=3)))
    # Assert
    assert panel.log == dict(ok=True, run=3)
    assert panel.last_log > 0


def test_on_log_status_with_bad_json_is_ignored(panel):
    # Arrange / Act
    panel.on_log_status(msg('nope'))
    # Assert
    assert panel.log == {} and panel.last_log == 0.0


def test_on_radar_counts_messages(panel):
    # Arrange / Act
    panel.on_radar(None)
    panel.on_radar(None)
    # Assert
    assert panel.radar_msgs == 2


# ----------------------------------------------------------------------- refresh_footer

def test_footer_shows_the_robot_while_its_status_is_fresh(panel):
    # Arrange
    panel.on_status(msg(dict(robot='B2 sim')))
    # Act
    panel.refresh_footer()
    # Assert
    assert 'Robot: B2 sim' in panel.footer.text()


def test_footer_robot_with_no_data_status_counts_as_no_data(panel):
    # Arrange
    panel.on_status(msg(dict(robot='no data')))
    # Act
    panel.refresh_footer()
    # Assert
    assert 'No data from the robot' in panel.footer.text()


def test_footer_robot_status_older_than_three_seconds_counts_as_no_data(panel):
    # Arrange
    panel.on_status(msg(dict(robot='B2 sim')))
    panel.last_status = time.monotonic() - 3.5
    # Act
    panel.refresh_footer()
    # Assert
    assert 'No data from the robot' in panel.footer.text()


def test_footer_radar_rate_is_updated_every_two_seconds(panel):
    # Arrange: 8 messages over the last 4 s
    panel.radar_msgs, panel.radar_t = 8, time.monotonic() - 4.0
    # Act
    panel.refresh_footer()
    # Assert
    assert 'Radar 2 Hz' in panel.footer.text()
    assert panel.radar_msgs == 0


def test_footer_radar_rate_is_kept_until_two_seconds_have_passed(panel):
    # Arrange: boundary, 1 s since the last update
    panel.radar_hz = 10.0
    panel.radar_msgs, panel.radar_t = 3, time.monotonic() - 1.0
    # Act
    panel.refresh_footer()
    # Assert
    assert 'Radar 10 Hz' in panel.footer.text()
    assert panel.radar_msgs == 3


def test_db_footer_postgres_ok_shows_the_alert_count(panel):
    # Arrange
    panel.on_log_status(msg(dict(ok=True, backend='postgres', where='rigsa@db', run=2,
                                 counts=dict(new=2, ack=1))))
    # Act
    panel.refresh_footer()
    # Assert
    text = panel.db_footer.text()
    assert 'PostgreSQL rigsa@db' in text and 'patrol run 2' in text
    assert 'alerts stored: 3' in text


def test_db_footer_sqlite_fallback_is_named(panel):
    # Arrange
    panel.on_log_status(msg(dict(ok=True, backend='sqlite', where='/tmp/e.db', run=1,
                                 counts={})))
    # Act
    panel.refresh_footer()
    # Assert
    assert 'SQLite (fallback)' in panel.db_footer.text()
    assert 'alerts stored: 0' in panel.db_footer.text()


def test_db_footer_unreachable_database_shows_the_error_and_pending_count(panel):
    # Arrange
    panel.on_log_status(msg(dict(ok=False, error='connection refused', pending=4)))
    # Act
    panel.refresh_footer()
    # Assert
    text = panel.db_footer.text()
    assert 'database unreachable (connection refused)' in text
    assert '4 alert(s) waiting' in text


def test_db_footer_old_log_status_counts_as_not_running(panel):
    # Arrange: boundary, status older than 3 s
    panel.on_log_status(msg(dict(ok=True, backend='postgres', where='x', run=1, counts={})))
    panel.last_log = time.monotonic() - 3.5
    # Act
    panel.refresh_footer()
    # Assert
    assert 'Event log not running' in panel.db_footer.text()


# ------------------------------------------------------------------------- reload_table

def fill(store, n, position=True):
    run = store.start_run('test')
    for i in range(1, n + 1):
        store.add_alert(run, db_alert(i, track='P%d' % i, position=position))
    return run


def test_reload_table_shows_the_alerts_of_the_current_run(panel, store):
    # Arrange
    panel.log = dict(run=fill(store, 1))
    # Act
    panel.reload_table()
    # Assert
    assert panel.table.rowCount() == 1
    texts = [panel.table.item(0, c).text() for c in range(4)]
    assert texts[0].startswith('#1  ')
    assert texts[1:] == ['P1', '3.1 m', 'NEW']


def test_reload_table_lists_the_newest_alert_first(panel, store):
    # Arrange
    panel.log = dict(run=fill(store, 3))
    # Act
    panel.reload_table()
    # Assert
    assert [panel.table.item(r, 1).text() for r in range(3)] == ['P3', 'P2', 'P1']


def test_reload_table_alert_without_a_position_says_no_position(panel, store):
    # Arrange
    panel.log = dict(run=fill(store, 1, position=False))
    # Act
    panel.reload_table()
    # Assert
    assert panel.table.item(0, 2).text() == 'no position'


def test_reload_table_shows_the_status_after_an_operator_action(panel, store):
    # Arrange
    run = fill(store, 1)
    store.add_action(run, 1, 'false_alarm', 'alice')
    panel.log = dict(run=run)
    # Act
    panel.reload_table()
    # Assert
    assert panel.table.item(0, 3).text() == 'FALSE ALARM'


def test_reload_table_shows_at_most_twelve_rows(panel, store):
    # Arrange: boundary, 13 alerts stored
    panel.log = dict(run=fill(store, 13))
    # Act
    panel.reload_table()
    # Assert
    assert panel.table.rowCount() == 12


def test_reload_table_with_no_alerts_is_empty(panel):
    # Arrange / Act
    panel.reload_table()
    # Assert
    assert panel.table.rowCount() == 0


def test_reload_table_keeps_the_old_rows_when_the_database_fails(panel, store):
    # Arrange
    panel.log = dict(run=fill(store, 2))
    panel.reload_table()
    store.close()
    # Act: the closed connection raises inside, which the panel must swallow
    panel.reload_table()
    # Assert
    assert panel.table.rowCount() == 2


# -------------------------------------------------------------------------------- spin

def test_spin_processes_ros_and_refreshes_the_display_at_most_every_half_second(
        panel, monkeypatch):
    # Arrange
    spins, reloads = [], []
    monkeypatch.setattr(rclpy, 'spin_once', lambda node, timeout_sec: spins.append(timeout_sec))
    monkeypatch.setattr(panel, 'reload_table', lambda: reloads.append(1))
    # Act
    panel.spin()
    panel.spin()
    # Assert: ROS runs every call, the display refresh only on the first
    assert spins == [0, 0]
    assert reloads == [1]


def test_spin_refreshes_again_after_half_a_second(panel, monkeypatch):
    # Arrange
    reloads = []
    monkeypatch.setattr(rclpy, 'spin_once', lambda node, timeout_sec: None)
    monkeypatch.setattr(panel, 'reload_table', lambda: reloads.append(1))
    panel.spin()
    panel.last_reload = time.monotonic() - 0.6
    # Act
    panel.spin()
    # Assert
    assert reloads == [1, 1]


def test_style_has_a_colour_and_title_for_every_alert_state():
    # Arrange / Act / Assert: show_alert looks up the state in STYLE, the table in STATUS_*
    assert set(operator_panel.STYLE) == {'new', 'ack', 'false_alarm', 'none'}
    assert set(operator_panel.STATUS_TEXT) == set(operator_panel.STATUS_COLOR) == {
        'new', 'ack', 'false_alarm'}
