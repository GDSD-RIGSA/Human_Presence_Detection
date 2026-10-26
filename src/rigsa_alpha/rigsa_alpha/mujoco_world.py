"""MuJoCo world used by the radar simulator: the building, walking people, and a copy of the B2.

The B2 here only mirrors the pose of the real simulation (unitree_mujoco, or the scripted robot): its base pose
comes from TF and its joint angles from /joint_states. It is used for drawing and is excluded from the radar rays.
With `b2_xml` pointing at unitree_mujoco's unitree_robots/b2/b2.xml the full B2 meshes are shown; otherwise a
simplified B2 with the same dimensions and joint names is built.

The radar is a ray-casting model of the TI AWR1443 (detection probabilities, noise, ghost points, doppler).
"""
import math
import os
import re

import mujoco
import numpy as np

WALL_H = 2.6
FURNITURE_H = 0.8
HIPS = {'FR': (0.3285, -0.072), 'FL': (0.3285, 0.072), 'RR': (-0.3285, -0.072), 'RL': (-0.3285, 0.072)}
THIGH, CALF = 0.35, 0.35
ROBOT_GROUPS = (2, 3)        # B2 geoms (visual 2, collision 3 in Unitree's b2.xml); excluded from radar rays
JOINTS = ['%s_%s_joint' % (leg, j) for leg in ('FR', 'FL', 'RR', 'RL') for j in ('hip', 'thigh', 'calf')]

# Radar model (AWR1443BOOST, indoor people-counting style config)
RADAR_RANGE = 10.0           # m
RADAR_HALF_AZ = 60.0         # deg
RADAR_HALF_EL = 15.0         # deg
RADAR_HZ = 10.0
AZ_STEP, EL_STEP = 1.5, 5.0  # ray grid, deg
P_DETECT = {'person': 0.6, 'wall': 0.035, 'furniture': 0.06, 'floor': 0.004}
SNR_DB = {'person': 16.0, 'wall': 11.0, 'furniture': 10.0, 'floor': 6.0}
NOISE_RANGE = 0.05           # m, 1 sigma
NOISE_AZ = 1.5               # deg
NOISE_EL = 4.0               # deg (the 1443 has poor elevation resolution)
GHOSTS_PER_FRAME = 1.0       # mean number of multipath ghost points


def building_geoms(walls, furniture, dx=0.0, dy=0.0):
    """MJCF geoms for walls and furniture, shifted by (dx, dy). Also used for the unitree_mujoco scene."""
    g = []
    for i, (x1, y1, x2, y2) in enumerate(walls):
        x1, y1, x2, y2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
        length, yaw = math.hypot(x2 - x1, y2 - y1), math.degrees(math.atan2(y2 - y1, x2 - x1))
        g.append('<geom name="wall%d" type="box" pos="%.3f %.3f %.3f" size="%.3f 0.05 %.3f" euler="0 0 %.2f" '
                 'rgba="0.82 0.82 0.80 1"/>' % (i, (x1 + x2) / 2, (y1 + y2) / 2, WALL_H / 2, length / 2 + 0.05,
                                                WALL_H / 2, yaw))
    for i, (x, y, w, h) in enumerate(furniture):
        g.append('<geom name="furniture%d" type="box" pos="%.2f %.2f %.2f" size="%.2f %.2f %.2f" rgba="0.55 0.42 0.3 1"/>'
                 % (i, x + dx + w / 2, y + dy + h / 2, FURNITURE_H / 2, w / 2, h / 2, FURNITURE_H / 2))
    return g


def simple_b2():
    """A B2 stand-in (no meshes) with Unitree's body/joint names, for when b2.xml is not available."""
    legs = []
    for leg, (hx, hy) in HIPS.items():
        side = 1 if hy > 0 else -1
        legs.append('''
      <body name="{l}_hip" pos="{hx} {hy} 0">
        <joint name="{l}_hip_joint" axis="1 0 0"/>
        <geom type="cylinder" size="0.055 0.04" euler="90 0 0" pos="0 {oy} 0" rgba="0.15 0.15 0.15 1" group="2"/>
        <body name="{l}_thigh" pos="0 {ty} 0">
          <joint name="{l}_thigh_joint" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -{T}" size="0.045" rgba="0.2 0.2 0.22 1" group="2"/>
          <body name="{l}_calf" pos="0 0 -{T}">
            <joint name="{l}_calf_joint" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 -{C}" size="0.03" rgba="0.3 0.3 0.32 1" group="2"/>
            <geom type="sphere" pos="0 0 -{C}" size="0.035" rgba="0.1 0.1 0.1 1" group="2"/>
          </body>
        </body>
      </body>'''.format(l=leg, hx=hx, hy=hy, oy=0.04 * side, ty=0.12 * side, T=THIGH, C=CALF))
    return '''
    <body name="base_link" pos="0 0 0.6">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.5 0.2 0.11" rgba="0.24 0.25 0.27 1" group="2"/>
      <geom type="box" pos="0.5 0 0.02" size="0.06 0.14 0.08" rgba="0.12 0.12 0.13 1" group="2"/>{legs}
    </body>'''.format(legs=''.join(legs))


def people_bodies(n):
    return ''.join('''
    <body name="person{i}" mocap="true" pos="0 0 -10">
      <geom name="person{i}_body" type="capsule" fromto="0 0 0.3 0 0 1.4" size="0.2" rgba="0.85 0.35 0.3 1"/>
      <geom name="person{i}_head" type="sphere" pos="0 0 1.62" size="0.11" rgba="0.9 0.75 0.6 1"/>
    </body>'''.format(i=i) for i in range(n))


def build_mjcf(walls, furniture, n_people, b2_xml=None):
    world = '\n    '.join(building_geoms(walls, furniture)) + people_bodies(n_people)
    extras = ('<visual><global offwidth="1280" offheight="720"/><headlight ambient="0.45 0.45 0.45"/></visual>\n'
              '  <asset><texture name="rigsa_grid" type="2d" builtin="checker" width="512" height="512" '
              'rgb1="0.75 0.75 0.72" rgb2="0.7 0.7 0.67"/>'
              '<material name="rigsa_floor" texture="rigsa_grid" texrepeat="30 30"/></asset>\n')
    floor = ('<light pos="0 0 12" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>'
             '<geom name="floor" type="plane" size="15 9 0.1" material="rigsa_floor"/>')
    if b2_xml and os.path.exists(b2_xml):
        # Unitree's full B2 model: make its mesh folder absolute and add our world into its worldbody
        xml = open(b2_xml).read()
        meshdir = os.path.join(os.path.dirname(os.path.abspath(b2_xml)), 'assets')
        xml = re.sub(r'meshdir="[^"]*"', 'meshdir="%s"' % meshdir, xml, count=1)
        xml = re.sub(r'<mujoco model="[^"]*"', '<mujoco model="RIGSA radar world"', xml, count=1)   # window title
        xml = xml.replace('<worldbody>', '<worldbody>\n    %s\n    %s' % (floor, world), 1)
        return xml.replace('</mujoco>', extras + '</mujoco>')
    return '<mujoco model="RIGSA radar world">\n  %s  <worldbody>\n    %s\n    %s%s\n  </worldbody>\n</mujoco>' % (
        extras, floor, world, simple_b2())


def multi_ray(m, d, origin, vecs, geomgroup, geomid, dist, n, cutoff):
    """mj_multiRay across MuJoCo versions (3.3+ added a `normal` argument)."""
    try:
        mujoco.mj_multiRay(m, d, origin, vecs, geomgroup, 1, -1, geomid, dist, None, n, cutoff)
    except TypeError:
        mujoco.mj_multiRay(m, d, origin, vecs, geomgroup, 1, -1, geomid, dist, n, cutoff)


class World:
    """Holds the MuJoCo model; people and the B2 copy are moved from outside, then the radar is scanned."""

    def __init__(self, walls, furniture, n_people, b2_xml=None, seed=None):
        self.full_b2 = bool(b2_xml and os.path.exists(b2_xml))
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(walls, furniture, n_people, b2_xml))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        m = self.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base_joint')
        self.root_q = m.jnt_qposadr[jid]
        self.joint_q = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS}
        self.person_mocap = [m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'person%d' % i)]
                             for i in range(n_people)]
        self.b2_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        for gid in range(m.ngeom):   # every B2 geom into a robot group, so the radar never sees its own robot
            if m.body_rootid[m.geom_bodyid[gid]] == self.b2_body and m.geom_group[gid] not in ROBOT_GROUPS:
                m.geom_group[gid] = 2   # group 2 is still drawn by the viewer (3 = collision shapes, hidden)
        self.geom_kind, self.geom_person = [], np.full(m.ngeom, -1)
        for gid in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
            kind = 'robot' if m.geom_group[gid] in ROBOT_GROUPS else 'other'
            for k in ('person', 'wall', 'furniture', 'floor'):
                if name.startswith(k):
                    kind = k
            if kind == 'person':
                self.geom_person[gid] = int(name[len('person'):].split('_')[0])
            self.geom_kind.append(kind)
        self.p_detect = np.array([P_DETECT.get(k, 0.0) for k in self.geom_kind])
        self.snr = np.array([SNR_DB.get(k, 0.0) for k in self.geom_kind])
        az = np.radians(np.arange(-RADAR_HALF_AZ, RADAR_HALF_AZ + 1e-6, AZ_STEP))
        el = np.radians(np.arange(-RADAR_HALF_EL, RADAR_HALF_EL + 1e-6, EL_STEP))
        self.ray_az, self.ray_el = [a.ravel() for a in np.meshgrid(az, el)]
        self.people_vel = np.zeros((n_people, 3))
        self.geomgroup = np.array([0 if g in ROBOT_GROUPS else 1 for g in range(6)], dtype=np.uint8)

    def set_robot(self, pos, quat_wxyz, joints):
        """Base pose in the map frame and joint angles {name: rad} (missing joints keep their value)."""
        q = self.data.qpos
        q[self.root_q:self.root_q + 7] = list(pos) + list(quat_wxyz)
        for name, val in joints.items():
            if name in self.joint_q:
                q[self.joint_q[name]] = val

    def set_people(self, people, people_vel):
        """people: list of (x, y, heading) or None; velocities (N, 3) in the map frame (for doppler)."""
        for i, p in enumerate(people):
            mid = self.person_mocap[i]
            if p is None:
                self.data.mocap_pos[mid] = [0, 0, -10]   # parked under the floor
            else:
                self.data.mocap_pos[mid] = [p[0], p[1], 0.0]
                self.data.mocap_quat[mid] = [math.cos(p[2] / 2), 0, 0, math.sin(p[2] / 2)]
        self.people_vel = np.asarray(people_vel, dtype=float).reshape(-1, 3)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def radar_scan(self, origin, rot, sensor_vel):
        """One radar frame from a radar at `origin` with rotation matrix `rot` (radar -> map), moving at
        `sensor_vel` (map frame). Returns (N, 5): x, y, z, snr_db, radial velocity in the radar frame."""
        m, d, rng = self.model, self.data, self.rng
        az = self.ray_az + rng.uniform(-0.5, 0.5, self.ray_az.size) * math.radians(AZ_STEP)
        el = self.ray_el + rng.uniform(-0.5, 0.5, self.ray_el.size) * math.radians(EL_STEP)
        local = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
        world = local @ np.asarray(rot).T
        n = len(world)
        geomid = np.full(n, -1, dtype=np.int32)
        dist = np.full(n, -1.0)
        multi_ray(m, d, np.asarray(origin, dtype=float), world.ravel(), self.geomgroup, geomid, dist, n, RADAR_RANGE)

        hit = (geomid >= 0) & (dist > 0.2) & (dist < RADAR_RANGE)
        hit &= rng.random(n) < np.where(geomid >= 0, self.p_detect[np.maximum(geomid, 0)], 0.0)
        idx = np.nonzero(hit)[0]
        r = dist[idx] + rng.normal(0, NOISE_RANGE, idx.size)
        a = az[idx] + np.radians(rng.normal(0, NOISE_AZ, idx.size))
        e = el[idx] + np.radians(rng.normal(0, NOISE_EL, idx.size))

        # radial velocity = (target velocity - sensor velocity) . ray direction (negative = approaching)
        tv = np.zeros((idx.size, 3))
        pid = self.geom_person[geomid[idx]]
        tv[pid >= 0] = self.people_vel[pid[pid >= 0]]
        vr = np.einsum('ij,ij->i', tv - np.asarray(sensor_vel, dtype=float), world[idx])
        snr = self.snr[geomid[idx]] + rng.normal(0, 2.0, idx.size)

        pts = np.stack([r * np.cos(e) * np.cos(a), r * np.cos(e) * np.sin(a), r * np.sin(e), snr, vr], axis=1)
        n_ghost = rng.poisson(GHOSTS_PER_FRAME)
        if n_ghost:
            gr = rng.uniform(1.0, RADAR_RANGE, n_ghost)
            ga = np.radians(rng.uniform(-RADAR_HALF_AZ, RADAR_HALF_AZ, n_ghost))
            ghosts = np.stack([gr * np.cos(ga), gr * np.sin(ga), rng.normal(0, 0.3, n_ghost),
                               rng.uniform(5, 9, n_ghost), rng.normal(0, 0.5, n_ghost)], axis=1)
            pts = np.vstack([pts, ghosts])
        return pts.astype(np.float32)

    def line_of_sight(self, origin, target):
        vec = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
        geomid = np.array([-1], dtype=np.int32)
        d = mujoco.mj_ray(self.model, self.data, np.asarray(origin, dtype=float), vec, self.geomgroup, 1, -1, geomid)
        return d < 0 or d > 1.0 or self.geom_kind[geomid[0]] == 'person'
