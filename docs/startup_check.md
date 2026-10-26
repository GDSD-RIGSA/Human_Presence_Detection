# Startup check (RIG-51)

This records a clean-clone build and test, as described in the README "Setup and startup" section.

## 26 Oct 2026, djole's laptop

- **Environment:** Ubuntu 22.04.5 LTS, ROS 2 Humble, Python 3.10.12, mujoco 3.3.6, numpy 1.21.5, scipy 1.8.0,
  pyserial 3.5.
- **Branch:** `RIG-38-djole-tasks` (on top of Eduardo's RIG-47/48 commit), cloned fresh into an empty folder.

| Step | Command | Result |
| --- | --- | --- |
| Build | `source /opt/ros/humble/setup.bash && colcon build` | 2 packages finished (b2_radar, rigsa_alpha) in 2.4 s. Only the stderr notes `byte-compiling is disabled`. |
| Executables | `ros2 pkg executables rigsa_alpha` / `b2_radar` | rigsa_alpha: detector, evaluator, make_unitree_scene, radar_driver, radar_sim, scripted_robot, site. b2_radar: b2_bridge, bag_radar. |
| Tests | `python3 -m pytest -q src/rigsa_alpha/test src/b2_radar/test` | 7 passed: 4 radar driver, 3 bag radar |
| Launch loads without the event log | import `alpha.launch.py` | loads, `HAVE_EVENT_LOG = False` |
| Run | `ROS_DOMAIN_ID=42 ros2 launch rigsa_alpha alpha.launch.py robot:=scripted rviz:=false` for 45 s | See below |

The run started site, scripted_robot, radar_sim, detector and evaluator, with no event logger or panel. It showed:

- radar sim: 10 Hz, 3 people
- `/radar/points` in frame `radar_link`
- 3 alerts (P1 at 2.3 m, P2 at 7.7 m, P3 at 6.4 m)
- the evaluator's result: 2/2 people detected, mean delay 0.44 s, 1 false alert
- all nodes finished cleanly on Ctrl+C

Not covered by this check:

- `robot:=unitree` (needs unitree_mujoco running)
- `robot:=none` with the real radar

The radar and domain 1 were busy with the live B2 demo at the time. On the same laptop and the same day, the
pre-repo version of this code ran with the real AWR1443 on the simulated B2: about 100 frames per 10 s, and alerts
were raised.
