"""Scores the detector against the simulation's ground truth (simulation only; it is not part of the product).

Subscribes /sim/people (true positions and whether the radar can see them), /tracks and /alerts.
Every `report_s` seconds it logs and publishes on /evaluation (JSON):
  people_seen      people who were visible to the radar for at least 1 s
  people_detected  of those, how many got a confirmed track within 1.0 m while visible
  mean_delay_s     time from first visible to the first matching alert
  false_alerts     alerts more than 1.5 m from every simulated person
"""
import json
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rigsa_alpha.common import spin

MATCH_M = 1.0
FALSE_M = 1.5


class Evaluator(Node):
    def __init__(self):
        super().__init__('rigsa_evaluator')
        self.declare_parameter('report_s', 20.0)
        self.truth = []
        self.visible_since = {}    # person -> first visible stamp in the current appearance
        self.appearances = []      # dicts: person, start, seen_s, detected_at
        self.current = {}          # person -> appearance dict
        self.false_alerts = 0
        self.alerts = 0
        self.create_subscription(String, '/sim/people', self.on_truth, 10)
        self.create_subscription(String, '/tracks', self.on_tracks, 10)
        self.create_subscription(String, '/alerts', self.on_alert, 10)
        self.pub = self.create_publisher(String, '/evaluation', 10)
        self.create_timer(self.get_parameter('report_s').value, self.report)

    def on_truth(self, msg):
        d = json.loads(msg.data)
        self.truth, t = d['people'], d['stamp']
        visible = {p['person'] for p in self.truth if p['visible']}
        for i in visible:
            ap = self.current.get(i)
            if ap is None:
                ap = self.current[i] = dict(person=i, start=t, last=t, detected_at=None)
                self.appearances.append(ap)
            ap['last'] = t
        for i in list(self.current):
            if i not in visible and t - self.current[i]['last'] > 2.0:
                del self.current[i]    # out of view for 2 s: the next sighting is a new appearance

    def on_tracks(self, msg):
        d = json.loads(msg.data)
        for tr in d['tracks']:
            for p in self.truth:
                ap = self.current.get(p['person'])
                if ap and ap['detected_at'] is None and math.hypot(tr['x'] - p['x'], tr['y'] - p['y']) < MATCH_M:
                    ap['detected_at'] = d['stamp']

    def on_alert(self, msg):
        a = json.loads(msg.data)
        self.alerts += 1
        if all(math.hypot(a['x'] - p['x'], a['y'] - p['y']) > FALSE_M for p in self.truth):
            self.false_alerts += 1
            self.get_logger().warn('Alert %s is not near any simulated person (false alert)' % a['event'])

    def report(self, publish=True):
        seen = [ap for ap in self.appearances if ap['last'] - ap['start'] >= 1.0]
        hit = [ap for ap in seen if ap['detected_at'] is not None]
        delays = [ap['detected_at'] - ap['start'] for ap in hit]
        res = dict(people_seen=len(seen), people_detected=len(hit), alerts=self.alerts, false_alerts=self.false_alerts,
                   mean_delay_s=round(sum(delays) / len(delays), 2) if delays else None)
        if publish:
            self.pub.publish(String(data=json.dumps(res)))
        self.get_logger().info('Evaluation: %d/%d people detected, mean delay %s s, %d alerts, %d false'
                               % (len(hit), len(seen), res['mean_delay_s'], self.alerts, self.false_alerts))


def main():
    rclpy.init()
    node = Evaluator()
    spin(node, lambda: node.report(publish=False))


if __name__ == '__main__':
    main()
