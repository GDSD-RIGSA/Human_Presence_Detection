"""Radar-only person detection, independent of ROS so it can be unit-tested.

Pipeline per radar frame (points already in the map frame):
  1. static background removal: drop points on or near walls and furniture in the occupancy map
  2. height and range gates
  3. accumulate the last few frames (radar point clouds are sparse) and cluster them (DBSCAN)
  4. track cluster centroids (nearest neighbour + smoothing); a track is confirmed after enough hits

Room-mode options for the real radar (all off by default, see params/room_improved.yaml):
  max_range, snr_db / snr_db_per_m      drop points beyond a range, or weaker than a range-dependent threshold
  min_points_per_m                      the cluster size needed to count changes with the cluster's range
  doppler_k                             DBSCAN on (x, y, doppler_k * radial velocity): separates people who walk
                                        past each other in different directions
  tracker='kalman'                      constant-velocity Kalman filter per track, Mahalanobis gating, Hungarian
                                        matching (scipy, greedy fallback)
  absorb_m                              clusters closer than this are merged, and a leftover cluster this close to
                                        a confirmed track is part of that person, not a new one
  new_track_min_points                  only clusters this strong start a new track (ghosts are weak)
  static_hold_s                         a confirmed person who stops (clutter removal hides them) is held in place
                                        with the same ID for this long, until points reappear within static_radius
"""
import math
from collections import deque

import numpy as np

# Tuning (metres, seconds). Documented in the README.
WALL_MARGIN = 0.35          # points closer than this to an occupied map cell are background
Z_MIN, Z_MAX = 0.15, 2.1    # person height band in the map frame
ACCUM_FRAMES = 4            # frames merged before clustering (0.4 s at 10 Hz)
EPS = 0.5                   # DBSCAN neighbourhood
MIN_POINTS = 5              # DBSCAN core size, over the accumulated frames
GATE = 1.2                  # max jump between a track and a new cluster per update
CONFIRM_HITS, CONFIRM_WINDOW = 5, 8   # confirmed after 5 hits in the last 8 updates
LOST_AFTER = 1.5            # track deleted when not seen for this long
REACQUIRE_S, REACQUIRE_M = 4.0, 2.0   # a new track near a just-lost confirmed one keeps its ID (no new alert)
COAST_S = 0.5               # a confirmed track not updated for this long can hand its ID to a new nearby track
MAX_SPEED = 2.0             # m/s, cap on a track's velocity estimate
PREDICT_S = 1.0             # predict pos + vel*dt only for gaps shorter than this; longer gaps use the last pos
I_COL, V_COL = 3, 4         # intensity (dB) and radial velocity (m/s) columns of an (N, 5) point array


class BackgroundMask:
    """Boolean grid: True where a point is static background (near occupied or outside the known map)."""

    def __init__(self, grid, resolution, origin, margin=WALL_MARGIN):
        occ = (grid != 0)            # occupied or unknown
        r = int(math.ceil(margin / resolution))
        h, w = occ.shape
        pad = np.pad(occ, r, constant_values=True)
        mask = np.zeros_like(occ)
        for dy in range(-r, r + 1):  # dilate with a disc
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    mask |= pad[r + dy:r + dy + h, r + dx:r + dx + w]
        self.mask, self.res, self.ox, self.oy = mask, resolution, origin[0], origin[1]

    def is_background(self, xy):
        c = np.floor((xy[:, 0] - self.ox) / self.res).astype(int)
        r = np.floor((xy[:, 1] - self.oy) / self.res).astype(int)
        h, w = self.mask.shape
        inside = (c >= 0) & (c < w) & (r >= 0) & (r < h)
        out = np.ones(len(xy), dtype=bool)
        out[inside] = self.mask[r[inside], c[inside]]
        return out


def dbscan(xy, eps=EPS, min_points=MIN_POINTS):
    """Small DBSCAN (O(n^2), fine for a few hundred radar points). Returns labels, -1 = noise."""
    n = len(xy)
    labels = np.full(n, -1)
    if n == 0:
        return labels
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    neigh = d2 <= eps * eps
    core = neigh.sum(1) >= min_points
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = cid
        stack = [i]
        while stack:
            j = stack.pop()
            if not core[j]:
                continue
            for k in np.nonzero(neigh[j] & (labels == -1))[0]:
                labels[k] = cid
                stack.append(k)
        cid += 1
    return labels


class Track:
    def __init__(self, tid, pos, t):
        self.id, self.pos, self.vel = tid, np.array(pos, dtype=float), np.zeros(2)
        self.first, self.last = t, t
        self.hits = deque([True], maxlen=CONFIRM_WINDOW)
        self.confirmed = False
        self.event = None        # alert number once confirmed
        self.state = 'new'       # new / ack / false_alarm (operator)
        self.n_points = 0
        self.trail = [tuple(pos)]
        self.life = 'moving'     # moving / static (room-mode static hold)
        self.hit_speed = 0.0     # speed estimate at the last update
        self.kf_x = np.array([pos[0], pos[1], 0.0, 0.0])   # Kalman state x, y, vx, vy (tracker='kalman')
        self.kf_P = None
        self.kf_t = t


class Detector:
    """Feed it radar frames in the map frame; it returns new alerts as tracks get confirmed."""

    def __init__(self, mask, gate=GATE, lost_after=LOST_AFTER, reacquire_s=REACQUIRE_S, reacquire_m=REACQUIRE_M,
                 eps=EPS, max_range=0.0, snr_db=0.0, snr_db_per_m=0.0, min_points_per_m=0.0, min_points_r0=2.0,
                 doppler_k=0.0, tracker='greedy', kf_accel=1.5, kf_meas=0.3, gate_chi2=9.21, absorb_m=0.0,
                 static_hold_s=0.0, static_speed=0.6, static_radius=1.5, new_track_min_points=0):
        """The keyword arguments override the tuning above (the real radar is sparser than the simulated one).
        The room-mode options (max_range ... static_radius) are off with their defaults; see the module docstring."""
        self.mask = mask
        self.gate, self.lost_after, self.eps = gate, lost_after, eps
        self.reacquire_s, self.reacquire_m = reacquire_s, reacquire_m
        self.max_range, self.snr_db, self.snr_db_per_m = max_range, snr_db, snr_db_per_m
        self.min_points_per_m, self.min_points_r0 = min_points_per_m, min_points_r0
        self.doppler_k, self.tracker, self.absorb_m = doppler_k, tracker, absorb_m
        self.kf_accel, self.kf_meas, self.gate_chi2 = kf_accel, kf_meas, gate_chi2
        self.static_hold_s, self.static_speed, self.static_radius = static_hold_s, static_speed, static_radius
        self.new_track_min_points = new_track_min_points
        self.frames = deque(maxlen=ACCUM_FRAMES)
        self.tracks = []
        self.lost = []           # recently lost confirmed tracks (for re-acquisition)
        self.next_id = 1
        self.next_event = 1
        self.clusters = []

    def filter(self, pts, sensor_xy=None):
        """pts: (N, 5) map-frame xyz + intensity + velocity. Returns the foreground points."""
        if len(pts) == 0:
            return pts
        keep = (pts[:, 2] > Z_MIN) & (pts[:, 2] < Z_MAX)
        pts = pts[keep]
        pts = pts[~self.mask.is_background(pts[:, :2])]
        if sensor_xy is not None and len(pts) and (self.max_range > 0 or self.snr_db > 0):
            r = np.hypot(pts[:, 0] - sensor_xy[0], pts[:, 1] - sensor_xy[1])
            keep = np.ones(len(pts), dtype=bool)
            if self.max_range > 0:
                keep &= r <= self.max_range
            if self.snr_db > 0:
                keep &= pts[:, I_COL] >= self.snr_db - self.snr_db_per_m * r
            pts = pts[keep]
        return pts

    def cluster(self, acc, sensor_xy=None):
        """DBSCAN over the merged frames. Returns [(centroid xy, n points, mean radial velocity)]."""
        if len(acc) == 0:
            return []
        feats = acc[:, :2]
        if self.doppler_k > 0:
            feats = np.column_stack([feats, self.doppler_k * acc[:, V_COL]])
        labels = dbscan(feats, eps=self.eps)
        clusters, vels = [], []
        for c in range(labels.max() + 1 if len(labels) else 0):
            sel = acc[labels == c]
            pos = sel[:, :2].mean(0)
            if self.min_points_per_m and sensor_xy is not None:
                r = np.hypot(pos[0] - sensor_xy[0], pos[1] - sensor_xy[1])
                if len(sel) < MIN_POINTS + self.min_points_per_m * (r - self.min_points_r0):
                    continue
            clusters.append((pos, len(sel)))
            vels.append(float(sel[:, V_COL].mean()))
        if self.absorb_m > 0:   # blobs this close together (and moving alike) are one person: a body often splits
            clusters = merge_close(clusters, self.absorb_m, vels, self.doppler_k)
        return clusters

    def update(self, pts_map, t, sensor_xy=None):
        """Returns a list of Track objects that were confirmed in this update (one alert each).
        pts_map: (N, 3) xyz or (N, 5) xyz + intensity + velocity in the map frame; sensor_xy: radar position in the
        map (needed by the range-dependent options)."""
        pts = np.asarray(pts_map, dtype=float)
        if pts.ndim != 2 or len(pts) == 0:
            pts = np.zeros((0, 5))
        elif pts.shape[1] < 5:   # xyz only: no intensity / velocity
            pts = np.column_stack([pts, np.zeros((len(pts), 5 - pts.shape[1]))])
        self.frames.append(self.filter(pts, sensor_xy))
        acc = np.vstack(list(self.frames))
        clusters = self.cluster(acc, sensor_xy)
        self.clusters = clusters

        if self.tracker == 'kalman':
            used_t, used_c = self.associate_kalman(clusters, t)
        else:
            used_t, used_c = self.associate_greedy(clusters, t)
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.hits.append(False)
        for ci, (pos, n) in enumerate(clusters):
            if ci in used_c:
                continue
            if self.absorb_m > 0 and any(o.confirmed and np.linalg.norm(o.pos - pos) < self.absorb_m
                                         for o in self.tracks):
                continue   # a second blob of a person already tracked, not a new person
            if n < self.new_track_min_points:
                continue   # weak blobs (wall reflections, sidelobes) may extend a track but never start one
            tr = Track(None, pos, t)
            tr.n_points = n
            if self.tracker == 'kalman':
                tr.kf_P = np.diag([self.kf_meas ** 2, self.kf_meas ** 2, 1.0, 1.0])
            self.tracks.append(tr)

        alive = []
        for tr in self.tracks:
            age = t - tr.last
            if (self.static_hold_s > 0 and tr.confirmed and age > COAST_S and tr.life == 'moving'
                    and tr.hit_speed < self.static_speed):
                tr.life = 'static'           # stopped: clutter removal hides a still person, so hold them in place
                tr.vel = np.zeros(2)
                tr.kf_x[2:] = 0.0
            if age <= self.lost_after or (tr.life == 'static' and age <= self.static_hold_s):
                alive.append(tr)
            elif tr.confirmed:
                self.lost.append((tr, t))
        self.tracks = alive
        self.lost = [(tr, tl) for tr, tl in self.lost if t - tl < self.reacquire_s]

        confirmed = []
        for tr in self.tracks:
            if not tr.confirmed and sum(tr.hits) >= CONFIRM_HITS:
                tr.confirmed = True
                old = self.reacquire(tr, t)
                if old is not None:      # same person seen again after a short gap: keep ID, no new alert
                    tr.id, tr.event, tr.state, tr.first = old.id, old.event, old.state, old.first
                    tr.trail = (old.trail + tr.trail)[-15:]
                    continue
                tr.id = 'P%d' % self.next_id
                tr.event = self.next_event
                self.next_id += 1
                self.next_event += 1
                confirmed.append(tr)
        return confirmed

    def associate_greedy(self, clusters, t):
        """Nearest neighbour, closest pairs first (the original tracker). Returns the used track / cluster indices."""
        pairs = sorted((self.distance(tr, c[0], t), ti, ci)
                       for ti, tr in enumerate(self.tracks) for ci, c in enumerate(clusters))
        used_t, used_c = set(), set()
        for d, ti, ci in pairs:
            if d > self.gate or ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            tr, (pos, n) = self.tracks[ti], clusters[ci]
            dt = max(t - tr.last, 1e-3)
            new_pos = 0.6 * pos + 0.4 * predict(tr, t)
            tr.vel = clamp_speed(0.7 * tr.vel + 0.3 * (new_pos - tr.pos) / dt)
            self.hit(tr, new_pos, n, t)
        return used_t, used_c

    def distance(self, tr, pos, t):
        """Greedy matching distance; a static track accepts points anywhere within static_radius."""
        d = np.linalg.norm(predict(tr, t) - pos)
        if tr.life == 'static':
            return d * self.gate / self.static_radius
        return d

    def hit(self, tr, pos, n, t):
        tr.pos, tr.last, tr.n_points = np.asarray(pos, dtype=float), t, n
        tr.hit_speed = float(np.linalg.norm(tr.vel))
        tr.life = 'moving'
        tr.hits.append(True)
        if np.linalg.norm(tr.pos - np.array(tr.trail[-1])) > 0.4:
            tr.trail = (tr.trail + [tuple(tr.pos)])[-15:]

    def kf_predict(self, tr, t):
        """Constant-velocity prediction to t (white acceleration noise kf_accel). A static track does not move, and a
        track unseen for PREDICT_S stops moving, so a lost person is not pushed away from where they reappear."""
        dt = t - tr.kf_t
        if dt <= 0:
            return
        if tr.life == 'static' or t - tr.last > PREDICT_S:
            tr.kf_x[2:] = 0.0
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        tr.kf_x = F @ tr.kf_x
        tr.kf_P = F @ tr.kf_P @ F.T + (self.kf_accel ** 2) * (G @ G.T)
        tr.kf_t = t
        tr.vel = clamp_speed(tr.kf_x[2:].copy())

    def associate_kalman(self, clusters, t):
        """Kalman predict, Mahalanobis gate (gate_chi2, 2 dof), Hungarian assignment, Kalman update."""
        H = np.eye(2, 4)
        R = np.eye(2) * self.kf_meas ** 2
        for tr in self.tracks:
            if tr.kf_P is None:
                tr.kf_x = np.array([tr.pos[0], tr.pos[1], tr.vel[0], tr.vel[1]], dtype=float)
                tr.kf_P = np.diag([self.kf_meas ** 2, self.kf_meas ** 2, 1.0, 1.0])
                tr.kf_t = tr.last
            self.kf_predict(tr, t)
        big = 1e9
        cost = np.full((len(self.tracks), len(clusters)), big)
        for ti, tr in enumerate(self.tracks):
            S_inv = np.linalg.inv(H @ tr.kf_P @ H.T + R)
            for ci, (pos, _) in enumerate(clusters):
                y = pos - tr.kf_x[:2]
                if tr.life == 'static':   # held in place: anywhere within static_radius counts
                    d2 = (np.linalg.norm(y) / self.static_radius) ** 2 * self.gate_chi2
                else:
                    d2 = float(y @ S_inv @ y)
                if d2 <= self.gate_chi2:
                    cost[ti, ci] = d2
        used_t, used_c = set(), set()
        for ti, ci in assign(cost, big):
            tr, (pos, n) = self.tracks[ti], clusters[ci]
            S = H @ tr.kf_P @ H.T + R
            K = tr.kf_P @ H.T @ np.linalg.inv(S)
            tr.kf_x = tr.kf_x + K @ (pos - tr.kf_x[:2])
            tr.kf_P = (np.eye(4) - K @ H) @ tr.kf_P
            tr.kf_x[2:] = clamp_speed(tr.kf_x[2:])
            tr.vel = tr.kf_x[2:].copy()
            self.hit(tr, tr.kf_x[:2].copy(), n, t)
            used_t.add(ti)
            used_c.add(ci)
        return used_t, used_c

    def reacquire(self, tr, t):
        """A recently lost confirmed track, or a confirmed one coasting (not updated for COAST_S), near tr. It is
        removed, so one person never shows as two tracks."""
        coasting = [o for o in self.tracks if o is not tr and o.confirmed and t - o.last > COAST_S]
        best, best_d = None, self.reacquire_m
        for old in [o for o, _ in self.lost if not any(o is k for k in self.tracks)] + coasting:
            d = np.linalg.norm(old.pos - tr.pos)
            if d < best_d:
                best, best_d = old, d
        if best is not None:
            self.lost = [(o, tl) for o, tl in self.lost if o is not best]
            self.tracks = [o for o in self.tracks if o is not best]
        return best

    def confirmed_tracks(self):
        return [tr for tr in self.tracks if tr.confirmed]

    def set_state(self, event, state):
        for tr in self.tracks + [o for o, _ in self.lost]:
            if tr.event == event:
                tr.state = state


def merge_close(clusters, dist, vels=None, doppler_k=0.0):
    """Merges clusters closer than dist (point-weighted centroid) until none are. With doppler_k, the distance also
    counts doppler_k * the difference of their mean radial velocities, so two people passing each other in
    opposite directions are not merged."""
    clusters = list(clusters)
    vels = list(vels) if vels is not None else [0.0] * len(clusters)
    merged = True
    while merged and len(clusters) > 1:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                (pi, ni), (pj, nj) = clusters[i], clusters[j]
                d = np.hypot(np.linalg.norm(pi - pj), doppler_k * (vels[i] - vels[j]))
                if d < dist:
                    clusters[i] = ((pi * ni + pj * nj) / (ni + nj), ni + nj)
                    vels[i] = (vels[i] * ni + vels[j] * nj) / (ni + nj)
                    del clusters[j], vels[j]
                    merged = True
                    break
            if merged:
                break
    return clusters


def assign(cost, big):
    """Minimum-cost one-to-one assignment, ignoring pairs that cost big. Hungarian (scipy) when available."""
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        return [(r, c) for r, c in zip(rows, cols) if cost[r, c] < big]
    except ImportError:   # greedy on the same costs
        out, used_r, used_c = [], set(), set()
        for r, c in sorted(zip(*np.nonzero(cost < big)), key=lambda rc: cost[rc]):
            if r not in used_r and c not in used_c:
                out.append((r, c))
                used_r.add(r)
                used_c.add(c)
        return out


def predict(tr, t):
    """Where tr should be at t: constant velocity for short gaps, the last position for long ones."""
    dt = t - tr.last
    return tr.pos + tr.vel * dt if dt < PREDICT_S else tr.pos


def clamp_speed(vel):
    speed = np.linalg.norm(vel)
    return vel * (MAX_SPEED / speed) if speed > MAX_SPEED else vel
