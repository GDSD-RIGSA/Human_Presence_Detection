"""`show_events` CLI: run listing, alert printing, filters, history and CSV export.

Everything runs against a throwaway SQLite event log (the CLI falls back to SQLite when Postgres
is unreachable), with `sys.argv` patched; output goes through capsys.
"""
import argparse
import csv
import os
import sqlite3
import sys
from datetime import datetime

import pytest

from rigsa_alpha import show_events
from rigsa_alpha.event_store import ALERT_COLUMNS, EventStore

T0 = '2026-10-28T10:00:00+02:00'
T1 = '2026-10-28T11:00:00+02:00'
T2 = '2026-10-28T12:00:00+02:00'
# port 1 is closed: the store falls back to SQLite, which is what the CLI test wants
NO_PG = 'host=127.0.0.1 port=1 dbname=rigsa user=rigsa password=rigsa connect_timeout=1'


@pytest.fixture
def db(tmp_path):
    """SQLite event log: an older `real` run, then the latest `sim` run with three alerts."""
    path = str(tmp_path / 'events.db')
    store = EventStore(NO_PG, path)
    assert store.backend == 'sqlite'
    old = store.start_run('real')
    run = store.start_run('sim')
    store.add_alert(old, dict(event=1, track='P9', stamp=T0, x=1.0, y=1.0, source='radar'))
    store.add_alert(run, dict(event=1, track='P1', stamp=T0, x=2.5, y=-1.0, distance=3.1,
                              bearing_deg=-20.0, n_points=9, robot_x=0.0, robot_y=0.0,
                              robot_yaw=0.1, source='radar'))
    store.add_alert(run, dict(event=2, track='P2', stamp=T1, source='radar'))   # no position
    store.add_alert(run, dict(event=3, track='P3', stamp=T2, x=-4.0, y=0.5, source='radar'))
    store.add_action(run, 3, 'ack', operator='ana')
    store.close()
    return path, run, old


def run_cli(monkeypatch, db, *argv, dsn=NO_PG):
    path = db[0]
    argv_list = ['show_events', '--dsn', dsn, '--sqlite', path] + [str(a) for a in argv]
    monkeypatch.setattr(sys, 'argv', argv_list)
    show_events.main()


# ---------------------------------------------------------------- formatting helpers

def test_fmt_prints_a_dash_for_missing_values():
    # Arrange / Act / Assert
    assert show_events.fmt(None, '%.1f') == '-'
    assert show_events.fmt(2.34, '%.1f') == '2.3'


def test_when_rejects_an_unparseable_datetime():
    # Arrange / Act / Assert
    with pytest.raises(argparse.ArgumentTypeError):
        show_events.when('yesterday')


def test_when_makes_a_naive_datetime_local():
    # Arrange / Act
    naive = show_events.when('2026-10-28 10:00')
    aware = show_events.when('2026-10-28T10:00:00+02:00')
    # Assert
    assert naive.tzinfo is not None
    assert aware.utcoffset().total_seconds() == 7200


# ---------------------------------------------------------------- run listing

def test_runs_lists_every_patrol_run_with_its_alert_count(db, monkeypatch, capsys):
    # Arrange / Act
    run_cli(monkeypatch, db, '--runs')
    out = capsys.readouterr().out
    # Assert: header, then newest run first, with the alert count per run
    assert 'run' in out and 'started' in out and 'alerts' in out
    lines = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith('Event log')]
    assert lines[1].strip().split()[-1] == '3'          # sim run (latest): three alerts
    assert lines[2].strip().split()[-1] == '1'          # real run (older): one alert


# ---------------------------------------------------------------- alert printing

def test_default_prints_the_latest_run_newest_first(db, monkeypatch, capsys):
    # Arrange / Act
    run_cli(monkeypatch, db)
    out = capsys.readouterr().out
    # Assert: latest run only, newest alert first, ack shown, missing position as '-'
    assert 'Patrol run %d' % db[1] in out
    assert 'P3' in out and 'P2' in out and 'P1' in out
    assert 'P9' not in out
    assert out.index('P3') < out.index('P1')
    assert 'ACK' in out
    assert 'missing' in out
    assert '-' in out.split('P2')[1].splitlines()[0]


def test_run_option_selects_one_run(db, monkeypatch, capsys):
    # Arrange / Act
    run_cli(monkeypatch, db, '--run', db[2])
    out = capsys.readouterr().out
    # Assert
    assert 'Patrol run %d' % db[2] in out
    assert 'P9' in out and 'P1' not in out


def test_all_runs_shows_alerts_from_every_run(db, monkeypatch, capsys):
    # Arrange / Act
    run_cli(monkeypatch, db, '--all-runs')
    out = capsys.readouterr().out
    # Assert: no single run header, both runs' tracks present
    assert 'Patrol run' not in out
    assert 'P9' in out and 'P1' in out


def test_limit_keeps_only_the_newest_rows(db, monkeypatch, capsys):
    # Arrange / Act
    run_cli(monkeypatch, db, '--limit', 1)
    out = capsys.readouterr().out
    # Assert
    assert 'P3' in out
    assert 'P1' not in out and 'P2' not in out


def test_status_and_position_filters(db, monkeypatch, capsys):
    # Arrange / Act
    run_cli(monkeypatch, db, '--status', 'ack')
    acked = capsys.readouterr().out
    run_cli(monkeypatch, db, '--position', 'missing')
    missing = capsys.readouterr().out
    # Assert
    assert 'P3' in acked and 'P1' not in acked
    assert 'P2' in missing and 'P1' not in missing


def test_since_and_until_select_a_time_window(db, monkeypatch, capsys):
    # Arrange / Act
    run_cli(monkeypatch, db, '--since', T1, '--until', T2)
    out = capsys.readouterr().out
    # Assert: T1 is inclusive, T2 exclusive
    assert 'P2' in out
    assert 'P1' not in out and 'P3' not in out


def test_no_match_is_reported(db, monkeypatch, capsys):
    # Arrange / Act
    run_cli(monkeypatch, db, '--status', 'false_alarm')
    out = capsys.readouterr().out
    # Assert
    assert '(no alerts match)' in out


# ---------------------------------------------------------------- operator history

def test_history_prints_the_operator_decisions(db, monkeypatch, capsys):
    # Arrange: rows are inserted oldest first, so the acked P3 alert is row 4
    run_cli(monkeypatch, db, '--history', 4)
    out = capsys.readouterr().out
    # Assert
    assert 'ACK' in out and 'ana' in out


def test_history_without_decisions_says_so(db, monkeypatch, capsys):
    # Arrange / Act
    run_cli(monkeypatch, db, '--history', 99)
    out = capsys.readouterr().out
    # Assert
    assert '(no operator decisions on alert row 99)' in out


# ---------------------------------------------------------------- CSV export

def test_csv_to_stdout_writes_the_header_and_rows(db, monkeypatch, capsys):
    # Arrange / Act
    run_cli(monkeypatch, db, '--csv', '-')
    captured = capsys.readouterr()
    # Assert: the status line moves to stderr, the CSV goes to stdout
    assert 'Event log: sqlite' in captured.err
    rows = list(csv.reader(captured.out.splitlines()))
    assert rows[0] == list(ALERT_COLUMNS)
    assert len(rows) == 4                                  # header + three alerts
    assert rows[1][ALERT_COLUMNS.index('track_id')] == 'P1'


def test_csv_to_file_reports_how_many_rows_were_written(db, monkeypatch, capsys, tmp_path):
    # Arrange
    target = tmp_path / 'alerts.csv'
    # Act
    run_cli(monkeypatch, db, '--csv', target)
    out = capsys.readouterr().out
    # Assert
    assert '3 alert(s) written to %s' % target in out
    rows = list(csv.reader(target.read_text().splitlines()))
    assert rows[0] == list(ALERT_COLUMNS)
    assert len(rows) == 4


# ---------------------------------------------------------------- failures

def test_an_empty_sqlite_file_without_schema_exits(tmp_path, monkeypatch, capsys):
    # Arrange: a SQLite file with no tables, opened with create=False
    path = str(tmp_path / 'empty.db')
    sqlite3.connect(path).close()
    monkeypatch.setattr(sys, 'argv', ['show_events', '--dsn', NO_PG, '--sqlite', path, '--runs'])
    # Act / Assert
    with pytest.raises(SystemExit) as err:
        show_events.main()
    assert 'Query failed' in str(err.value.code)


def test_an_unwritable_sqlite_location_exits_before_querying(tmp_path, monkeypatch):
    # Arrange: the parent path is a file, so neither Postgres nor SQLite can be opened
    blocker = tmp_path / 'blocker'
    blocker.write_text('x')
    monkeypatch.setattr(sys, 'argv', ['show_events', '--dsn', NO_PG,
                                      '--sqlite', str(blocker / 'events.db'), '--runs'])
    # Act / Assert
    with pytest.raises((SystemExit, OSError)):
        show_events.main()


def test_detected_at_values_are_stored_as_iso_strings(db):
    # Arrange / Act: what the CLI prints comes straight from SQLite
    store = EventStore(NO_PG, db[0], create=False)
    rows = store.find_alerts(run_id=db[1])
    store.close()
    # Assert
    assert [r['track_id'] for r in rows] == ['P1', 'P2', 'P3']
    assert datetime.fromisoformat(rows[0]['detected_at']) == datetime.fromisoformat(T0)
    assert os.path.basename(db[0]) == 'events.db'
