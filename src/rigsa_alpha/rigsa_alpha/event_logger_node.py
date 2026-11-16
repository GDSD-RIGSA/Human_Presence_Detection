import json
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rigsa_alpha.common import spin
from rigsa_alpha.event_store import DEFAULT_DSN, DEFAULT_SQLITE, BufferedWriter, EventStore

UNSAVED_PATH = os.path.expanduser('~/.ros/rigsa_unsaved_events.jsonl')


class EventLogger(Node):
    def __init__(self):
        super().__init__('rigsa_event_logger')
        self.declare_parameter('db_dsn', DEFAULT_DSN)
        self.declare_parameter('sqlite_fallback', True)
        self.declare_parameter('sqlite_path', DEFAULT_SQLITE)
        self.declare_parameter('source', 'sim')
        self.declare_parameter('max_pending', 1000)
        self.declare_parameter('retry_s', 5.0)
        p = self.get_parameter
        self.store = EventStore(p('db_dsn').value, p('sqlite_path').value,
                                fallback=p('sqlite_fallback').value)
        if self.store.backend == 'postgres':
            self.get_logger().info('Event log: PostgreSQL %s' % self.store.where)
        else:
            self.get_logger().warn('PostgreSQL not reachable (%s). Event log falls back to '
                                   'SQLite %s' % (self.store.error, self.store.where))
        self.run_id = self.store.start_run(p('source').value)
        self.get_logger().info('Patrol run %d started' % self.run_id)
        self.writer = BufferedWriter(self.store, self.run_id, max_pending=p('max_pending').value,
                                     retry_s=p('retry_s').value)
        self.create_subscription(String, '/alerts', self.on_alert, 10)
        self.create_subscription(String, '/alerts/ack', self.on_ack, 10)
        self.status_pub = self.create_publisher(String, '/event_log/status', 10)
        self.create_timer(1.0, self.publish_status)

    def log(self, lines):
        for level, text in lines:
            getattr(self.get_logger(), level)(text)

    def on_alert(self, msg):
        try:
            a = json.loads(msg.data)
        except ValueError as e:
            self.get_logger().error('Rejected alert: not JSON (%s)' % e)
            return
        self.log(self.writer.submit_alert(a))

    def on_ack(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError as e:
            self.get_logger().error('Rejected operator action: not JSON (%s)' % e)
            return
        self.log(self.writer.submit_action(d))

    def publish_status(self):
        self.log(self.writer.flush())
        counts = {}
        if self.writer.db_ok:
            try:
                counts = self.store.counts(self.run_id)
            except Exception as e:
                self.get_logger().warn('Event log status query failed: %s' % e,
                                       throttle_duration_sec=10.0)
        w = self.writer
        self.status_pub.publish(String(data=json.dumps(dict(
            backend=self.store.backend, where=self.store.where, ok=w.db_ok, run=self.run_id,
            counts=counts, pending=len(w.pending), stored=w.stored, rejected=w.rejected,
            dropped=w.dropped, error=w.last_error))))

    def close(self):
        try:
            self.log(self.writer.flush(force=True))
        except Exception:
            pass
        n = self.writer.dump_pending(UNSAVED_PATH)
        if n:
            print('[event_logger] %d event(s) never reached the database; saved to %s'
                  % (n, UNSAVED_PATH), flush=True)
        try:
            self.store.end_run(self.run_id)
        except Exception:
            pass
        self.store.close()


def main():
    rclpy.init()
    node = EventLogger()
    spin(node, node.close)


if __name__ == '__main__':
    main()
