\echo '== Patrol runs, newest first, with alert counts'
SELECT r.id, r.started_at, r.ended_at, r.source, COUNT(a.id) AS alerts
FROM patrol_runs r LEFT JOIN alerts a ON a.run_id = r.id
GROUP BY r.id ORDER BY r.id DESC LIMIT 10;

\echo '== Alerts of the latest run, oldest first'
SELECT id, event_no, track_id, detected_at, stored_at, round(x_m::numeric, 2) AS x, round(y_m::numeric, 2) AS y,
       round(distance_m::numeric, 1) AS dist, position_status, n_points, status
FROM alerts WHERE run_id = (SELECT max(id) FROM patrol_runs) ORDER BY detected_at, id;

\echo '== Alerts in the last hour (any run)'
SELECT id, run_id, event_no, track_id, detected_at, position_status, status
FROM alerts WHERE detected_at >= now() - interval '1 hour' ORDER BY detected_at;

\echo '== Alerts still waiting for the operator'
SELECT id, run_id, event_no, track_id, detected_at FROM alerts WHERE status = 'new' ORDER BY detected_at;

\echo '== Alerts stored without a full position (RIG-58)'
SELECT id, run_id, event_no, track_id, detected_at, position_status, x_m, y_m, robot_x
FROM alerts WHERE position_status <> 'ok' ORDER BY detected_at;

\echo '== Operator decisions with their alert'
SELECT o.acted_at, o.operator, o.action, a.run_id, a.event_no, a.track_id,
       o.acted_at - a.detected_at AS reaction_time
FROM operator_actions o JOIN alerts a ON a.id = o.alert_id ORDER BY o.acted_at DESC LIMIT 20;

\echo '== Per run: alerts by status and storage delay (stored_at - detected_at)'
SELECT run_id, status, COUNT(*) AS n, max(stored_at - detected_at) AS max_store_delay
FROM alerts GROUP BY run_id, status ORDER BY run_id DESC, status;

\echo '== Integrity checks (all must return 0)'
SELECT 'ok rows without position' AS check, COUNT(*) FROM alerts
  WHERE position_status = 'ok' AND (x_m IS NULL OR y_m IS NULL)
UNION ALL SELECT 'missing rows with a position', COUNT(*) FROM alerts
  WHERE position_status = 'missing' AND (x_m IS NOT NULL OR y_m IS NOT NULL)
UNION ALL SELECT 'handled alerts without a decision', COUNT(*) FROM alerts a
  WHERE a.status <> 'new' AND NOT EXISTS (SELECT 1 FROM operator_actions o WHERE o.alert_id = a.id);
