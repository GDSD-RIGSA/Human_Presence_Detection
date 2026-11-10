import json
import math
import os
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone

SCHEMA_SQLITE = '''
CREATE TABLE IF NOT EXISTS patrol_runs (id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL, ended_at TEXT, robot TEXT NOT NULL DEFAULT 'B2',
    source TEXT NOT NULL, notes TEXT);
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES patrol_runs(id),
    event_no INTEGER NOT NULL, track_id TEXT NOT NULL, detected_at TEXT NOT NULL,
    stored_at TEXT NOT NULL,
    x_m REAL, y_m REAL, distance_m REAL, bearing_deg REAL,
    position_status TEXT NOT NULL DEFAULT 'ok'
    CHECK (position_status IN ('ok', 'no_robot_pose', 'missing')), n_points INTEGER,
    robot_x REAL, robot_y REAL,
    robot_yaw REAL, source TEXT NOT NULL DEFAULT 'radar', status TEXT NOT NULL DEFAULT 'new'
    CHECK (status IN ('new', 'ack', 'false_alarm')), handled_at TEXT, UNIQUE (run_id, event_no));
CREATE TABLE IF NOT EXISTS operator_actions (id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER NOT NULL REFERENCES alerts(id),
    action TEXT NOT NULL CHECK (action IN ('ack', 'false_alarm')), acted_at TEXT NOT NULL,
    operator TEXT);
CREATE INDEX IF NOT EXISTS alerts_detected_at ON alerts (detected_at DESC);
'''

DEFAULT_DSN = 'host=localhost port=5432 dbname=rigsa user=rigsa password=rigsa'
DEFAULT_SQLITE = os.path.expanduser('~/.ros/rigsa_alpha_events.db')
POSITION_STATUSES = ('ok', 'no_robot_pose', 'missing')
ACTIONS = ('ack', 'false_alarm')
ALERT_COLUMNS = ('id', 'run_id', 'event_no', 'track_id', 'detected_at', 'stored_at', 'x_m',
                 'y_m', 'distance_m', 'bearing_deg', 'position_status', 'n_points', 'robot_x',
                 'robot_y', 'robot_yaw', 'source', 'status', 'handled_at')


def schema_postgres():
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('rigsa_alpha'), 'sql', 'schema.sql')
    except Exception:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sql', 'schema.sql')
    with open(path) as f:
        return f.read()


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()


def _number(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def normalize_alert(a):
    if not isinstance(a, dict):
        raise ValueError('alert is not a JSON object')
    warnings = []
    try:
        event = int(a['event'])
    except (KeyError, TypeError, ValueError):
        raise ValueError('alert has no valid event number')
    track = a.get('track')
    if track is None or str(track).strip() == '':
        raise ValueError('alert %d has no track id' % event)
    stamp = a.get('stamp')
    try:
        detected = datetime.fromisoformat(str(stamp))
        if detected.tzinfo is None:
            detected = detected.astimezone()
        detected = detected.isoformat()
    except (TypeError, ValueError):
        detected = now_iso()
        warnings.append('alert %d has no valid timestamp (%r); receive time used' % (event, stamp))

    x, y = _number(a.get('x')), _number(a.get('y'))
    robot = [_number(a.get(k)) for k in ('robot_x', 'robot_y', 'robot_yaw')]
    dist, bearing = _number(a.get('distance')), _number(a.get('bearing_deg'))
    if x is None or y is None:
        status, x, y, dist, bearing = 'missing', None, None, None, None
        warnings.append(
            'alert %d has no person position; stored with position_status=missing' % event)
    elif None in robot:
        status, dist, bearing, robot = 'no_robot_pose', None, None, [None, None, None]
        warnings.append(
            'alert %d has no robot pose; stored with position_status=no_robot_pose' % event)
    else:
        status = 'ok'
        if dist is None or bearing is None:
            dx, dy = x - robot[0], y - robot[1]
            dist = math.hypot(dx, dy)
            bearing = (math.degrees(math.atan2(dy, dx) - robot[2]) + 180.0) % 360.0 - 180.0
    n_points = a.get('n_points')
    try:
        n_points = int(n_points) if n_points is not None else None
    except (TypeError, ValueError):
        n_points = None
    row = dict(event_no=event, track_id=str(track), detected_at=detected, x_m=x, y_m=y,
               distance_m=dist, bearing_deg=bearing, position_status=status, n_points=n_points,
               robot_x=robot[0], robot_y=robot[1], robot_yaw=robot[2],
               source=str(a.get('source') or 'radar'))
    return row, warnings


def is_connection_error(e):
    try:
        import psycopg2
        if isinstance(e, (psycopg2.OperationalError, psycopg2.InterfaceError)):
            return True
    except ImportError:
        pass
    if isinstance(e, sqlite3.OperationalError):
        return True
    return isinstance(e, (ConnectionError, TimeoutError, OSError))


class EventStore:

    def __init__(self, dsn=DEFAULT_DSN, sqlite_path=DEFAULT_SQLITE, fallback=True, create=True,
                 connect_timeout=3):
        self.error = None
        self.dsn, self.connect_timeout = dsn, connect_timeout
        try:
            self._connect_postgres()
            self.backend = 'postgres'
            self.where = '%s@%s' % (self.conn.info.dbname, self.conn.info.host)
            if create:
                with self.conn.cursor() as c:
                    c.execute(schema_postgres())
        except Exception as e:
            if not fallback:
                raise
            self.error = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            os.makedirs(os.path.dirname(sqlite_path) or '.', exist_ok=True)
            if create:
                self._retire_old_sqlite(sqlite_path)
            self.conn = sqlite3.connect(sqlite_path, check_same_thread=False, timeout=2)
            self.backend = 'sqlite'
            self.where = sqlite_path
            if create:
                self.conn.executescript(SCHEMA_SQLITE)
                self.conn.commit()

    def _connect_postgres(self):
        import psycopg2
        self.conn = psycopg2.connect(self.dsn, connect_timeout=self.connect_timeout)
        self.conn.autocommit = True

    @staticmethod
    def _retire_old_sqlite(path):
        if not os.path.exists(path):
            return
        con = sqlite3.connect(path)
        try:
            cols = [r[1] for r in con.execute('PRAGMA table_info(alerts)')]
        finally:
            con.close()
        if cols and 'position_status' not in cols:
            os.replace(path, path + '.v03.bak')

    def connected(self):
        if self.backend == 'postgres':
            return self.conn.closed == 0
        return True

    def reconnect(self):
        if self.backend != 'postgres':
            return
        try:
            self.conn.close()
        except Exception:
            pass
        self._connect_postgres()

    def _q(self, sql):
        return sql if self.backend == 'postgres' else sql.replace('%s', '?')

    def execute(self, sql, args=(), fetch=False):
        cur = self.conn.cursor()
        try:
            cur.execute(self._q(sql), args)
            rows = cur.fetchall() if fetch else None
            if self.backend == 'sqlite':
                self.conn.commit()
            return rows
        except Exception:
            if self.backend == 'sqlite':
                self.conn.rollback()
            raise
        finally:
            cur.close()

    def now(self):
        return now_iso()

    def start_run(self, source, notes=None):
        if self.backend == 'postgres':
            return self.execute(
                'INSERT INTO patrol_runs (source, notes) VALUES (%s, %s) RETURNING id',
                (source, notes), fetch=True)[0][0]
        self.execute('INSERT INTO patrol_runs (started_at, source, notes) VALUES (%s, %s, %s)',
                     (self.now(), source, notes))
        return self.execute('SELECT last_insert_rowid()', fetch=True)[0][0]

    def end_run(self, run_id):
        self.execute('UPDATE patrol_runs SET ended_at = %s WHERE id = %s', (self.now(), run_id))

    def add_alert(self, run_id, a):
        row, _ = normalize_alert(a)
        return self.add_alert_row(run_id, row)

    def add_alert_row(self, run_id, row):
        cols = ('run_id', 'event_no', 'track_id', 'detected_at', 'stored_at', 'x_m', 'y_m',
                'distance_m', 'bearing_deg', 'position_status', 'n_points', 'robot_x', 'robot_y',
                'robot_yaw', 'source')
        values = dict(row, run_id=run_id, stored_at=self.now())
        args = tuple(values[c] for c in cols)
        sql = ('INSERT INTO alerts (%s) VALUES (%s) ON CONFLICT (run_id, event_no) DO NOTHING'
               % (', '.join(cols), ', '.join(['%s'] * len(cols))))
        if self.backend == 'postgres':
            rows = self.execute(sql + ' RETURNING id', args, fetch=True)
            return rows[0][0] if rows else None
        before = self.conn.total_changes
        self.execute(sql, args)
        if self.conn.total_changes == before:
            return None
        return self.execute('SELECT last_insert_rowid()', fetch=True)[0][0]

    def add_action(self, run_id, event_no, action, operator=None):
        if action not in ACTIONS:
            raise ValueError('unknown operator action %r' % (action,))
        rows = self.execute('SELECT id FROM alerts WHERE run_id = %s AND event_no = %s',
                            (run_id, event_no), fetch=True)
        if not rows:
            return None
        alert_id, now = rows[0][0], self.now()
        self.execute('INSERT INTO operator_actions (alert_id, action, acted_at, operator) '
                     'VALUES (%s, %s, %s, %s)',
                     (alert_id, action, now, operator))
        self.execute('UPDATE alerts SET status = %s, handled_at = %s WHERE id = %s',
                     (action, now, alert_id))
        return alert_id

    def latest_run(self):
        rows = self.execute('SELECT id FROM patrol_runs ORDER BY id DESC LIMIT 1', fetch=True)
        return rows[0][0] if rows else None

    def runs(self, limit=10):
        return self.execute('SELECT r.id, r.started_at, r.ended_at, r.source, COUNT(a.id) '
                            'FROM patrol_runs r LEFT JOIN alerts a ON a.run_id = r.id '
                            'GROUP BY r.id ORDER BY r.id DESC LIMIT %d' % int(limit), fetch=True)

    def recent_alerts(self, run_id=None, limit=12):
        sql = ('SELECT id, run_id, event_no, track_id, detected_at, distance_m, bearing_deg, '
               'status FROM alerts')
        args = ()
        if run_id is not None:
            sql += ' WHERE run_id = %s'
            args = (run_id,)
        return self.execute(sql + ' ORDER BY id DESC LIMIT %d' % int(limit), args, fetch=True)

    def find_alerts(self, run_id=None, since=None, until=None, status=None, position_status=None,
                    limit=1000):
        where, args = [], []
        for col, op, val in (('run_id', '=', run_id), ('detected_at', '>=', since),
                             ('detected_at', '<', until), ('status', '=', status),
                             ('position_status', '=', position_status)):
            if val is None:
                continue
            if col == 'detected_at' and self.backend == 'sqlite':
                val = val.isoformat() if isinstance(val, datetime) else str(val)
                col = 'julianday(detected_at)'
                where.append('%s %s julianday(%%s)' % (col, op))
            else:
                where.append('%s %s %%s' % (col, op))
            args.append(val)
        sql = 'SELECT %s FROM alerts' % ', '.join(ALERT_COLUMNS)
        if where:
            sql += ' WHERE ' + ' AND '.join(where)
        sql += ' ORDER BY detected_at, id LIMIT %d' % int(limit)
        return [dict(zip(ALERT_COLUMNS, r)) for r in self.execute(sql, tuple(args), fetch=True)]

    def alert_history(self, alert_id):
        return self.execute('SELECT action, acted_at, operator FROM operator_actions '
                            'WHERE alert_id = %s ORDER BY acted_at, id', (alert_id,), fetch=True)

    def counts(self, run_id):
        rows = self.execute('SELECT status, COUNT(*) FROM alerts WHERE run_id = %s '
                            'GROUP BY status', (run_id,), fetch=True)
        return {s: n for s, n in rows}

    def position_counts(self, run_id):
        rows = self.execute('SELECT position_status, COUNT(*) FROM alerts WHERE run_id = %s '
                            'GROUP BY position_status',
                            (run_id,), fetch=True)
        return {s: n for s, n in rows}

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


class BufferedWriter:

    def __init__(self, store, run_id, max_pending=1000, retry_s=5.0, clock=time.monotonic):
        self.store, self.run_id = store, run_id
        self.pending = deque()
        self.max_pending, self.retry_s, self.clock = max_pending, retry_s, clock
        self.next_try = 0.0
        self.db_ok = True
        self.last_error = None
        self.stored = self.rejected = self.dropped = self.duplicates = 0

    def submit_alert(self, a):
        try:
            row, warnings = normalize_alert(a)
        except ValueError as e:
            self.rejected += 1
            return [('error', 'Rejected alert: %s' % e)]
        return [('warn', w) for w in warnings] + self._submit(('alert', row))

    def submit_action(self, d):
        try:
            item = ('action', dict(event=int(d['event']), action=str(d['action']),
                                   operator=d.get('operator')))
            if item[1]['action'] not in ACTIONS:
                raise ValueError('unknown action %r' % item[1]['action'])
        except (KeyError, TypeError, ValueError) as e:
            self.rejected += 1
            return [('error', 'Rejected operator action %r: %s' % (d, e))]
        return self._submit(item)

    def _submit(self, item):
        logs = []
        if len(self.pending) >= self.max_pending:
            self.pending.popleft()
            self.dropped += 1
            logs.append(('error', 'Event log buffer full (%d); oldest unsaved event dropped'
                         % self.max_pending))
        self.pending.append(item)
        return logs + self.flush(force=self.db_ok)

    def flush(self, force=False):
        logs = []
        if not self.pending:
            return logs
        if not self.db_ok and not force and self.clock() < self.next_try:
            return logs
        if not self.db_ok or not self.store.connected():
            try:
                self.store.reconnect()
            except Exception as e:
                return logs + self._down(e, during='reconnect')
            logs.append(('info', 'Event log: database connection restored; '
                                 'writing %d waiting event(s)' % len(self.pending)))
        while self.pending:
            kind, data = self.pending[0]
            try:
                if kind == 'alert':
                    row_id = self.store.add_alert_row(self.run_id, data)
                else:
                    row_id = self.store.add_action(self.run_id, data['event'], data['action'],
                                                   data['operator'])
            except Exception as e:
                if is_connection_error(e):
                    return logs + self._down(e, during='write')
                self.pending.popleft()
                self.rejected += 1
                logs.append(('error', 'Database refused %s %s: %s' % (
                    kind, data.get('event_no', data.get('event')),
                    str(e).strip().splitlines()[0])))
                continue
            self.pending.popleft()
            self.db_ok, self.last_error = True, None
            if kind == 'alert':
                if row_id is None:
                    self.duplicates += 1
                    logs.append(('warn', 'Alert %d was already stored; duplicate ignored'
                                 % data['event_no']))
                else:
                    self.stored += 1
                    logs.append(('info', 'Stored alert %d (person %s, position %s) as row %d in %s'
                                 % (data['event_no'], data['track_id'], data['position_status'],
                                    row_id, self.store.backend)))
            elif row_id is None:
                logs.append(('warn', 'Operator action on alert %d ignored: '
                                     'alert not in the event log' % data['event']))
            else:
                logs.append(('info', 'Stored operator action %s on alert %d'
                             % (data['action'], data['event'])))
        return logs

    def _down(self, e, during):
        first = self.db_ok
        self.db_ok = False
        self.last_error = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        self.next_try = self.clock() + self.retry_s
        if first:
            return [('error', 'Event log database unreachable (%s failed: %s). '
                              '%d event(s) kept in memory; retrying every %.0f s'
                              % (during, self.last_error, len(self.pending), self.retry_s))]
        return []

    def dump_pending(self, path):
        if not self.pending:
            return 0
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'a') as f:
            for kind, data in self.pending:
                f.write(json.dumps(dict(kind=kind, run_id=self.run_id, **data)) + '\n')
        return len(self.pending)
