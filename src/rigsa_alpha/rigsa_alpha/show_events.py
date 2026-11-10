import argparse
import csv
import sys
from datetime import datetime

from rigsa_alpha.event_store import (ALERT_COLUMNS, DEFAULT_DSN, DEFAULT_SQLITE, POSITION_STATUSES,
                                     EventStore)

STATUS = {'new': 'NEW', 'ack': 'ACK', 'false_alarm': 'FALSE ALARM'}


def fmt(v, spec):
    return spec % v if v is not None else '-'


def when(text):
    try:
        t = datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError('use a date/time like "2026-10-28 10:00"')
    return t if t.tzinfo else t.astimezone()


def main():
    ap = argparse.ArgumentParser(description='Prints alerts from the Event Log')
    ap.add_argument('--dsn', default=DEFAULT_DSN)
    ap.add_argument('--sqlite', default=DEFAULT_SQLITE)
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--all-runs', action='store_true')
    ap.add_argument('--run', type=int)
    ap.add_argument('--runs', action='store_true', help='list patrol runs')
    ap.add_argument('--since', type=when)
    ap.add_argument('--until', type=when)
    ap.add_argument('--status', choices=sorted(STATUS))
    ap.add_argument('--position', choices=POSITION_STATUSES)
    ap.add_argument('--history', type=int, metavar='ROW',
                    help='operator decisions on one alert row')
    ap.add_argument('--csv', metavar='FILE',
                    help="write the selected alerts to a CSV file ('-' = stdout)")
    args, _ = ap.parse_known_args()
    try:
        store = EventStore(args.dsn, args.sqlite, fallback=True, create=False)
    except Exception as e:
        sys.exit('Event log not reachable: %s' % e)
    print('Event log: %s %s' % (store.backend, store.where),
          file=sys.stderr if args.csv == '-' else sys.stdout)
    try:
        if args.runs:
            print('%4s  %-25s %-25s %-7s %s' % ('run', 'started', 'ended', 'source', 'alerts'))
            for rid, start, end, source, n in store.runs(args.limit):
                print('%4d  %-25s %-25s %-7s %d'
                      % (rid, str(start)[:25], str(end or '-')[:25], source, n))
            return
        if args.history is not None:
            rows = store.alert_history(args.history)
            for action, at, operator in rows:
                print('%-25s %-12s %s' % (str(at)[:25], STATUS[action], operator or '-'))
            if not rows:
                print('(no operator decisions on alert row %d)' % args.history)
            return
        run = args.run
        if run is None and not args.all_runs:
            run = store.latest_run()
        rows = store.find_alerts(run_id=run, since=args.since, until=args.until,
                                 status=args.status, position_status=args.position, limit=100000)
        if not args.csv:
            rows = rows[-args.limit:]
        if args.csv:
            out = sys.stdout if args.csv == '-' else open(args.csv, 'w', newline='')
            w = csv.writer(out)
            w.writerow(ALERT_COLUMNS)
            for r in rows:
                w.writerow([r[c] for c in ALERT_COLUMNS])
            if out is not sys.stdout:
                out.close()
                print('%d alert(s) written to %s' % (len(rows), args.csv))
            return
        if run is not None:
            print('Patrol run %d' % run)
        print('%6s %4s %4s %-7s %-25s %7s %7s %8s %-13s %s' % (
            'row', 'run', 'no', 'person', 'detected at', 'x m', 'y m', 'dist m', 'position',
            'status'))
        for r in reversed(rows):
            print('%6d %4d %4d %-7s %-25s %7s %7s %8s %-13s %s' % (
                r['id'], r['run_id'], r['event_no'], r['track_id'], str(r['detected_at'])[:25],
                fmt(r['x_m'], '%.1f'), fmt(r['y_m'], '%.1f'), fmt(r['distance_m'], '%.1f'),
                r['position_status'], STATUS[r['status']]))
        if not rows:
            print('(no alerts match)')
    except Exception as e:
        sys.exit('Query failed: %s' % str(e).strip().splitlines()[0])
    finally:
        store.close()


if __name__ == '__main__':
    main()
