import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from rigsa_alpha.event_store import DEFAULT_DSN, BufferedWriter, EventStore, normalize_alert

PG_DSN = os.environ.get('RIGSA_TEST_DSN', DEFAULT_DSN)
TEST_SCHEMA = 'rigsa_test'
T0 = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone(timedelta(hours=1)))


def pg_available():
    try:
        import psycopg2
        psycopg2.connect(PG_DSN, connect_timeout=2).close()
        return True
    except Exception:
        return False


def pg_admin(sql):
    import psycopg2
    con = psycopg2.connect(PG_DSN, connect_timeout=2)
    con.autocommit = True
    with con.cursor() as c:
        c.execute(sql)
    con.close()


@pytest.fixture(params=['sqlite', pytest.param('postgres', marks=pytest.mark.skipif(
    not pg_available(), reason='PostgreSQL not reachable (set RIGSA_TEST_DSN)'))])
def store(request, tmp_path):
    if request.param == 'sqlite':
        s = EventStore('host=127.0.0.1 port=1 connect_timeout=1', str(tmp_path / 'events.db'))
        assert s.backend == 'sqlite'
        yield s
        s.close()
        return
    pg_admin('DROP SCHEMA IF EXISTS %s CASCADE; CREATE SCHEMA %s' % (TEST_SCHEMA, TEST_SCHEMA))
    s = EventStore(PG_DSN + ' options=-csearch_path=%s' % TEST_SCHEMA, fallback=False)
    assert s.backend == 'postgres'
    yield s
    s.close()
    pg_admin('DROP SCHEMA IF EXISTS %s CASCADE' % TEST_SCHEMA)


def alert(event, track='P1', t=T0, **kw):
    a = dict(event=event, track=track, stamp=t.isoformat(), x=2.5, y=-1.0, distance=3.1,
             bearing_deg=-20.0, n_points=9, robot_x=0.0, robot_y=0.5, robot_yaw=0.1,
             source='radar')
    a.update(kw)
    return {k: v for k, v in a.items() if v is not None}


def as_dt(v):
    return v if isinstance(v, datetime) else datetime.fromisoformat(v)


def test_tables_and_columns_exist(store):
    if store.backend == 'postgres':
        rows = store.execute("SELECT table_name, column_name FROM information_schema.columns "
                             "WHERE table_schema = current_schema()", fetch=True)
    else:
        rows = [(t, r[1]) for t in ('patrol_runs', 'alerts', 'operator_actions')
                for r in store.execute('PRAGMA table_info(%s)' % t, fetch=True)]
    cols = {}
    for table, col in rows:
        cols.setdefault(table, set()).add(col)
    assert {'id', 'started_at', 'ended_at', 'source'} <= cols['patrol_runs']
    assert {'run_id', 'event_no', 'track_id', 'detected_at', 'stored_at', 'x_m', 'y_m',
            'distance_m', 'bearing_deg', 'position_status', 'robot_x', 'robot_y', 'robot_yaw',
            'status', 'handled_at'} <= cols['alerts']
    assert {'alert_id', 'action', 'acted_at', 'operator'} <= cols['operator_actions']


def test_schema_rejects_invalid_values(store):
    run = store.start_run('sim')
    store.add_alert(run, alert(1))
    with pytest.raises(Exception):
        store.execute("UPDATE alerts SET status = 'maybe' WHERE run_id = %s", (run,))
    with pytest.raises(Exception):
        store.execute("UPDATE alerts SET position_status = 'guess' WHERE run_id = %s", (run,))


def test_insert_and_read_back(store):
    run = store.start_run('real', notes='test')
    row_id = store.add_alert(run, alert(1))
    assert row_id is not None
    [a] = store.find_alerts(run_id=run)
    assert (a['id'], a['event_no'], a['track_id'], a['status'], a['position_status']) == (
        row_id, 1, 'P1', 'new', 'ok')
    assert a['x_m'] == pytest.approx(2.5) and a['y_m'] == pytest.approx(-1.0)
    assert a['distance_m'] == pytest.approx(3.1) and a['n_points'] == 9 and a['source'] == 'radar'


def test_duplicate_event_is_ignored(store):
    run = store.start_run('sim')
    assert store.add_alert(run, alert(1)) is not None
    assert store.add_alert(run, alert(1, track='P9')) is None
    assert len(store.find_alerts(run_id=run)) == 1


def test_same_event_number_in_two_runs(store):
    r1, r2 = store.start_run('sim'), store.start_run('sim')
    assert store.add_alert(r1, alert(1)) != store.add_alert(r2, alert(1))


def test_operator_action_updates_status_and_history(store):
    run = store.start_run('sim')
    row_id = store.add_alert(run, alert(1))
    assert store.add_action(run, 1, 'ack', 'djole') == row_id
    [a] = store.find_alerts(run_id=run)
    assert a['status'] == 'ack' and a['handled_at'] is not None
    [(action, _at, operator)] = store.alert_history(row_id)
    assert (action, operator) == ('ack', 'djole')
    assert store.add_action(run, 99, 'ack') is None
    with pytest.raises(ValueError):
        store.add_action(run, 1, 'delete')


def test_timestamps_are_kept(store):
    run = store.start_run('sim')
    store.add_alert(run, alert(1, t=T0))
    [a] = store.find_alerts(run_id=run)
    assert as_dt(a['detected_at']) == T0
    stored = as_dt(a['stored_at'])
    assert abs((stored - datetime.now(timezone.utc)).total_seconds()) < 60


def test_missing_person_position_is_explicit(store):
    run = store.start_run('sim')
    store.add_alert(run, alert(1, x=None, y=None, distance=None, bearing_deg=None))
    store.add_alert(run, alert(2, x=float('nan')))
    for a in store.find_alerts(run_id=run):
        assert a['position_status'] == 'missing'
        assert a['x_m'] is None and a['y_m'] is None
        assert a['distance_m'] is None and a['bearing_deg'] is None
    assert store.position_counts(run) == {'missing': 2}


def test_missing_robot_pose_is_explicit(store):
    run = store.start_run('sim')
    store.add_alert(run, alert(1, robot_x=None, robot_y=None, robot_yaw=None))
    [a] = store.find_alerts(run_id=run)
    assert a['position_status'] == 'no_robot_pose'
    assert a['x_m'] == pytest.approx(2.5)
    assert a['robot_x'] is None and a['distance_m'] is None and a['bearing_deg'] is None


def test_panel_query_handles_missing_position(store):
    run = store.start_run('sim')
    store.add_alert(run, alert(1, x=None, y=None))
    [(_id, _run, no, track, _t, dist, bearing, status)] = store.recent_alerts(run_id=run)
    assert (no, dist, bearing, status) == (1, None, None, 'new')


def test_normalize_alert():
    row, warnings = normalize_alert(alert(1))
    assert row['position_status'] == 'ok' and warnings == []
    row, warnings = normalize_alert(alert(2, distance=None, bearing_deg=None, robot_yaw=0.0))
    assert row['distance_m'] == pytest.approx(((2.5) ** 2 + 1.5 ** 2) ** 0.5)
    row, warnings = normalize_alert(alert(3, stamp=None))
    assert row['detected_at'] and 'timestamp' in warnings[0]
    for bad in ({}, {'event': 'x', 'track': 'P1'}, {'event': 4}, [1, 2]):
        with pytest.raises(ValueError):
            normalize_alert(bad)


def test_retrieval_queries(store):
    r1 = store.start_run('sim')
    for k in range(5):
        store.add_alert(r1, alert(k + 1, track='P%d' % (k + 1), t=T0 + timedelta(minutes=k)))
    store.add_action(r1, 2, 'ack')
    store.add_action(r1, 3, 'false_alarm')
    r2 = store.start_run('real')
    store.add_alert(r2, alert(1, t=T0 + timedelta(hours=1), x=None))

    assert [a['event_no'] for a in store.find_alerts(run_id=r1)] == [1, 2, 3, 4, 5]
    window = store.find_alerts(since=T0 + timedelta(minutes=1), until=T0 + timedelta(minutes=3))
    assert [a['event_no'] for a in window] == [2, 3]
    utc = (T0 + timedelta(minutes=4)).astimezone(timezone.utc)
    assert [a['event_no'] for a in store.find_alerts(run_id=r1, since=utc)] == [5]
    assert [a['event_no'] for a in store.find_alerts(run_id=r1, status='new')] == [1, 4, 5]
    assert [a['run_id'] for a in store.find_alerts(position_status='missing')] == [r2]
    assert store.counts(r1) == {'new': 3, 'ack': 1, 'false_alarm': 1}
    assert store.latest_run() == r2
    assert [(r[0], r[3], r[4]) for r in store.runs()] == [(r2, 'real', 1), (r1, 'sim', 5)]
    assert [r[2] for r in store.recent_alerts(run_id=r1, limit=2)] == [5, 4]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def break_database(store, monkeypatch):
    if store.backend == 'postgres':
        import psycopg2
        store.conn.close()
        monkeypatch.setattr(store, 'dsn', 'host=127.0.0.1 port=1 dbname=rigsa user=rigsa')
        monkeypatch.setattr(store, 'connect_timeout', 1)
        return psycopg2.OperationalError

    def down(*_a, **_k):
        raise sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(store, 'execute', down)
    return sqlite3.OperationalError


def test_database_outage_keeps_events_in_order(store, monkeypatch):
    run = store.start_run('sim')
    clock = Clock()
    w = BufferedWriter(store, run, retry_s=5.0, clock=clock)
    w.submit_alert(alert(1))
    assert w.stored == 1 and w.db_ok

    with monkeypatch.context() as m:
        break_database(store, m)
        logs = w.submit_alert(alert(2, track='P2'))
        assert not w.db_ok and len(w.pending) == 1
        assert any(level == 'error' and 'unreachable' in text for level, text in logs)
        w.submit_action(dict(event=2, action='ack', operator='djole'))
        w.submit_alert(alert(3, track='P3'))
        assert len(w.pending) == 3
        clock.t = 1.0
        assert w.flush() == []
        clock.t = 6.0
        w.flush()
        assert len(w.pending) == 3

    clock.t = 12.0
    logs = w.flush()
    assert w.db_ok and not w.pending and w.stored == 3
    assert any('restored' in text for _l, text in logs)
    rows = store.find_alerts(run_id=run)
    assert [(a['event_no'], a['status']) for a in rows] == [(1, 'new'), (2, 'ack'), (3, 'new')]


def test_bad_events_are_rejected_not_retried(store):
    run = store.start_run('sim')
    w = BufferedWriter(store, run)
    logs = w.submit_alert({'track': 'P1'})
    assert w.rejected == 1 and not w.pending and logs[0][0] == 'error'
    logs = w.submit_action({'event': 1, 'action': 'delete'})
    assert w.rejected == 2 and not w.pending
    logs = w.submit_action({'event': 7, 'action': 'ack'})
    assert any('not in the event log' in text for _l, text in logs)
    w.submit_alert(alert(1))
    w.submit_alert(alert(1))
    assert (w.stored, w.duplicates) == (1, 1)
    assert w.db_ok


def test_buffer_limit_and_dump(store, monkeypatch, tmp_path):
    run = store.start_run('sim')
    w = BufferedWriter(store, run, max_pending=3, clock=Clock())
    break_database(store, monkeypatch)
    for k in range(5):
        w.submit_alert(alert(k + 1))
    assert len(w.pending) == 3 and w.dropped == 2
    assert [d['event_no'] for _k, d in w.pending] == [3, 4, 5]
    path = tmp_path / 'unsaved.jsonl'
    assert w.dump_pending(str(path)) == 3
    assert len(path.read_text().splitlines()) == 3


def test_fallback_to_sqlite_when_postgres_down(tmp_path):
    s = EventStore('host=127.0.0.1 port=1 dbname=rigsa user=rigsa', str(tmp_path / 'f.db'),
                   connect_timeout=1)
    assert s.backend == 'sqlite' and s.error
    with pytest.raises(Exception):
        EventStore('host=127.0.0.1 port=1 dbname=rigsa user=rigsa', fallback=False,
                   connect_timeout=1)


def test_old_sqlite_file_is_set_aside(tmp_path):
    path = tmp_path / 'old.db'
    con = sqlite3.connect(str(path))
    con.execute('CREATE TABLE alerts (id INTEGER PRIMARY KEY, x_m REAL NOT NULL)')
    con.commit()
    con.close()
    s = EventStore('host=127.0.0.1 port=1 connect_timeout=1', str(path))
    run = s.start_run('sim')
    s.add_alert(run, alert(1, x=None))
    assert (tmp_path / 'old.db.v03.bak').exists()
