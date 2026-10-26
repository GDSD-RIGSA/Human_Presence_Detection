"""The scripted alpha scenario (map frame, metres): three people walk around the building.

With robot:=unitree the B2 (unitree_mujoco) stands guard at SENTRY_POST. With robot:=scripted it patrols
the hallway instead. Only the simulation nodes use this file. The detector never sees it; the evaluator uses the people's
positions (published as ground truth on /sim/people) to score the detections.
"""
import math

SCENARIO_S = 80.0      # the scene repeats every 80 s
PATROL_SPEED = 0.35    # m/s, B2
TURN_S = 4.0           # s, the scripted B2 turns on the spot at each end of the patrol
PATROL = [(-10.0, -0.8), (10.0, -0.8)]   # robot:=scripted, back and forth along the hallway
SENTRY_POST = (-2.0, 0.0, 0.0)            # robot:=unitree: where the unitree_mujoco world origin sits in the map (x, y, yaw)
WALK = 0.8             # m/s, people

PEOPLE = [
    # leaves the meeting room and walks to the entrance
    dict(start=4.0, path=[(-1.2, 5.0), (0.8, 3.2), (0.8, 2.0), (3.0, 0.8), (13.5, 0.4)]),
    # comes out of the workshop, walks down the hallway into the office
    dict(start=18.0, path=[(8.6, -6.4), (6.4, -4.6), (5.5, -2.5), (4.0, -1.0), (2.0, 1.6), (-9.4, 1.6), (-9.4, 2.6),
                           (-10.5, 5.0)]),
    # comes in through the entrance and goes into the workshop
    dict(start=48.0, path=[(13.5, -0.5), (7.0, -0.6), (5.5, -1.5), (5.5, -4.0), (3.0, -6.0)]),
]


def along(path, dist):
    """(x, y, heading) at distance `dist` along a polyline, or None before its start / after its end."""
    if dist < 0:
        return None
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        seg = math.hypot(x2 - x1, y2 - y1)
        if dist <= seg:
            t = dist / seg
            return x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, math.atan2(y2 - y1, x2 - x1)
        dist -= seg
    return None


def robot_pose(t):
    """B2 pose (x, y, yaw) at time t (s since start); the patrol does not reset with the scene.
    Walks from PATROL[0] to PATROL[1], turns on the spot, walks back, turns, and so on."""
    (x1, y1), (x2, y2) = PATROL
    walk = math.hypot(x2 - x1, y2 - y1) / PATROL_SPEED
    yaw = math.atan2(y2 - y1, x2 - x1)
    d = t % (2 * (walk + TURN_S))
    for (ax, ay), (bx, by), heading in (((x1, y1), (x2, y2), yaw), ((x2, y2), (x1, y1), yaw + math.pi)):
        if d < walk:
            f = d / walk
            return ax + (bx - ax) * f, ay + (by - ay) * f, heading
        d -= walk
        if d < TURN_S:
            return bx, by, heading + math.pi * d / TURN_S
        d -= TURN_S


def people_poses(t):
    """List of (x, y, heading) or None per person at scene time t."""
    t = t % SCENARIO_S
    return [along(p['path'], (t - p['start']) * WALK) for p in PEOPLE]
