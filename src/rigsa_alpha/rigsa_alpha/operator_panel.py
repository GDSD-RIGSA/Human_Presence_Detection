import json
import signal
import sys
import time
from datetime import datetime

import rclpy
from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtGui import QBrush, QColor, QFont
from python_qt_binding.QtWidgets import (QAbstractItemView, QApplication, QFrame, QHBoxLayout, QHeaderView, QLabel,
                                         QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from rigsa_alpha.event_store import DEFAULT_DSN, DEFAULT_SQLITE, EventStore

STYLE = {
    'new': ('#c62828', 'Person detected'),
    'ack': ('#2e7d32', 'Acknowledged'),
    'false_alarm': ('#616161', 'Marked as false alarm'),
    'none': ('#90a4ae', 'No active alerts'),
}
STATUS_TEXT = {'new': 'NEW', 'ack': 'ACK', 'false_alarm': 'FALSE ALARM'}
STATUS_COLOR = {'new': '#d32f2f', 'ack': '#2e7d32', 'false_alarm': '#757575'}


class OperatorPanel(QWidget):
    def __init__(self, node, store, operator):
        super().__init__()
        self.node = node
        self.store = store
        self.operator = operator
        self.current = None
        self.last_status = 0.0
        self.last_log = 0.0
        self.log = {}
        self.ack_pub = node.create_publisher(String, '/alerts/ack', 10)
        node.create_subscription(String, '/alerts', self.on_alert, 10)
        node.create_subscription(String, '/operator/status', self.on_status, 10)
        self.radar_msgs, self.radar_hz, self.radar_t = 0, 0.0, time.monotonic()
        node.create_subscription(PointCloud2, '/radar/points', self.on_radar, qos_profile_sensor_data)
        node.create_subscription(String, '/event_log/status', self.on_log_status, 10)
        node.create_subscription(String, '/alerts/ack', self.on_ack, 10)

        self.setWindowTitle('RIGSA Operator – Alerts')
        self.resize(600, 900)
        self.setFont(QFont('Sans', 12))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(14)

        self.alert = QFrame()
        al = QVBoxLayout(self.alert)
        al.setContentsMargins(20, 16, 20, 16)
        self.alert_title = QLabel()
        self.alert_title.setFont(QFont('Sans', 22, QFont.Bold))
        self.alert_detail = QLabel()
        self.alert_detail.setFont(QFont('Sans', 16))
        self.alert_time = QLabel()
        for w in (self.alert_title, self.alert_detail, self.alert_time):
            w.setStyleSheet('color: white; background: transparent;')
            al.addWidget(w)
        lay.addWidget(self.alert)

        row = QHBoxLayout()
        self.ack_btn = QPushButton('✓  Acknowledge')
        self.fa_btn = QPushButton('✕  False alarm')
        self.ack_btn.setStyleSheet('QPushButton { background: #2a74b3; color: white; font-weight: bold; padding: 14px; '
                                   'font-size: 17px; border-radius: 4px; } QPushButton:disabled { background: #9fb8cf; }')
        self.fa_btn.setStyleSheet('QPushButton { padding: 14px; font-size: 17px; }')
        self.ack_btn.clicked.connect(lambda: self.handle('ack'))
        self.fa_btn.clicked.connect(lambda: self.handle('false_alarm'))
        row.addWidget(self.ack_btn)
        row.addWidget(self.fa_btn)
        lay.addLayout(row)

        head = QLabel('Recent events (from the event log database)')
        head.setFont(QFont('Sans', 14, QFont.Bold))
        lay.addWidget(head)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Time', 'Person', 'Distance', 'Status'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        lay.addWidget(self.table, 1)

        self.footer = QLabel()
        lay.addWidget(self.footer)
        self.db_footer = QLabel()
        lay.addWidget(self.db_footer)
        self.show_alert()
        self.refresh_footer()

    def on_alert(self, msg):
        try:
            a = json.loads(msg.data)
        except ValueError:
            return
        a['status'] = 'new'
        self.current = a
        self.show_alert()
        QApplication.beep()

    def on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
        except ValueError:
            return
        self.last_status = time.monotonic()

    def on_radar(self, _msg):
        self.radar_msgs += 1

    def on_ack(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        if self.current and self.current['event'] == d.get('event') and self.current['status'] == 'new':
            self.current['status'] = d.get('action', 'ack')
            self.show_alert()

    def on_log_status(self, msg):
        try:
            self.log = json.loads(msg.data)
        except ValueError:
            return
        self.last_log = time.monotonic()

    def handle(self, action):
        if not self.current or self.current['status'] != 'new':
            return
        self.current['status'] = action
        self.ack_pub.publish(String(data=json.dumps(dict(event=self.current['event'], action=action,
                                                         operator=self.operator))))
        self.show_alert()

    def show_alert(self):
        state = self.current['status'] if self.current else 'none'
        color, title = STYLE[state]
        self.alert.setStyleSheet('QFrame { background: %s; border-radius: 8px; }' % color)
        self.alert_title.setText(('⚠  ' if state == 'new' else '') + title)
        if self.current:
            a = self.current
            if a.get('distance') is not None and a.get('direction'):
                self.alert_detail.setText('%s · %.1f m · %s' % (a['track'], a['distance'], a['direction']))
            else:
                self.alert_detail.setText('%s · position unknown' % a['track'])
            self.alert_time.setText(a['time'])
        else:
            self.alert_detail.setText('Waiting for the radar…')
            self.alert_time.setText('')
        self.ack_btn.setEnabled(state == 'new')
        self.fa_btn.setEnabled(state == 'new')

    def reload_table(self):
        try:
            rows = self.store.recent_alerts(run_id=self.log.get('run'), limit=12)
        except Exception:
            return
        self.table.setRowCount(len(rows))
        for r, (_id, _run, event_no, track, detected, dist, _bearing, status) in enumerate(rows):
            if isinstance(detected, str):
                detected = datetime.fromisoformat(detected)
            t = detected.astimezone().strftime('%H:%M:%S')
            where = '%.1f m' % dist if dist is not None else 'no position'
            for c, text in enumerate(('#%d  %s' % (event_no, t), track, where, STATUS_TEXT[status])):
                item = QTableWidgetItem(text)
                if c == 3:
                    item.setForeground(QBrush(QColor('white')))
                    item.setBackground(QBrush(QColor(STATUS_COLOR[status])))
                    item.setTextAlignment(Qt.AlignCenter)
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                self.table.setItem(r, c, item)

    def refresh_footer(self):
        now = time.monotonic()
        if now - self.radar_t >= 2.0:
            self.radar_hz, self.radar_msgs, self.radar_t = self.radar_msgs / (now - self.radar_t), 0, now
        radar = '&nbsp;&nbsp; Radar %.0f Hz' % self.radar_hz
        if now - self.last_status < 3.0 and self.status.get('robot') != 'no data':
            self.footer.setText('<span style="color:#2e7d32">●</span> Robot: %s %s' % (self.status['robot'], radar))
        else:
            self.footer.setText('<span style="color:#c62828">●</span> No data from the robot %s' % radar)
        if time.monotonic() - self.last_log < 3.0 and self.log.get('ok'):
            n = sum(self.log.get('counts', {}).values())
            name = 'PostgreSQL' if self.log['backend'] == 'postgres' else 'SQLite (fallback)'
            self.db_footer.setText('<span style="color:#2e7d32">●</span> Event log: %s %s &nbsp;&nbsp; patrol run %s '
                                   '&nbsp;&nbsp; alerts stored: %d' % (name, self.log['where'], self.log['run'], n))
        elif time.monotonic() - self.last_log < 3.0:
            self.db_footer.setText('<span style="color:#c62828">●</span> Event log: database unreachable (%s), '
                                   '%d alert(s) waiting to be saved' % (self.log.get('error'), self.log.get('pending', 0)))
        else:
            self.db_footer.setText('<span style="color:#c62828">●</span> Event log not running')

    def spin(self):
        rclpy.spin_once(self.node, timeout_sec=0)
        now = time.monotonic()
        if now - getattr(self, 'last_reload', 0.0) > 0.5:
            self.last_reload = now
            self.refresh_footer()
            self.reload_table()


def main():
    rclpy.init(args=sys.argv)
    node = rclpy.create_node('rigsa_operator_panel')
    node.declare_parameter('db_dsn', DEFAULT_DSN)
    node.declare_parameter('sqlite_path', DEFAULT_SQLITE)
    node.declare_parameter('operator', 'operator')
    p = node.get_parameter
    store = EventStore(p('db_dsn').value, p('sqlite_path').value, fallback=True, create=False)
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    panel = OperatorPanel(node, store, p('operator').value)
    panel.move(1310, 0)
    panel.show()
    timer = QTimer()
    timer.timeout.connect(panel.spin)
    timer.start(50)
    code = app.exec_()
    store.close()
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
