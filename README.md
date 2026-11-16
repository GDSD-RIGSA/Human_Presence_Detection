# Human_Presence_Detection
Using mmWave Radar in Unitree B2, we aim to develop a mobile human presence detection platform.

# Testing

## Docker (no ROS 2 install needed)

`scripts/docker_test.sh` builds `docker/Dockerfile` (ROS 2 Humble), starts a throwaway PostgreSQL 14 on a private network, then runs `colcon build` and `colcon test` for `b2_radar` and `rigsa_alpha`. Run it from bash or WSL:

```bash
./scripts/docker_test.sh                                  # about 1 min with a cached image
./scripts/docker_test.sh --packages-select rigsa_alpha    # extra arguments go to colcon test
```

The repo is mounted read-only and built in a container-local copy, so no `build/`, `install/` or `log/` appears in the tree. The image has pytest, numpy, scipy, psycopg2, pyserial, rosbag2, Qt (`python_qt_binding`, used offscreen) and MuJoCo `3.17.0`.

If the script fails with `cannot execute: required file not found`, the checkout has CRLF line endings (Windows with `core.autocrlf=true`). `.gitattributes` forces LF for `*.sh`; re-checkout the file with `git rm --cached scripts/docker_test.sh && git checkout -- scripts/docker_test.sh`.

## Ubuntu 22.04 with ROS 2 Humble

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select b2_radar rigsa_alpha
source install/setup.bash
colcon test --packages-select b2_radar rigsa_alpha --event-handlers console_direct+
colcon test-result --verbose
```

Without MuJoCo or `python_qt_binding`, `src/rigsa_alpha/test/conftest.py` leaves out the three test files that need them, so a plain setup runs 229 tests instead of 339.

## Continuous integration

`.github/workflows/tests.yml` runs two jobs on every push and pull request: `lint` (`ament_flake8 src`) and `test` (colcon build and test against a `postgres:14` service, with the same Qt and MuJoCo packages). Current state: 339 tests, 0 failures, 0 skipped.

# Software Implementation Tasks
Tasks per Sprint - Current Progress

## Sprint 4
- PB09 - Done and Merged to Dev in PR2
- PB11 - Done
- PB12 - Done
- PB17 - Done

## Sprint 5
- PB08 - Hardware Mounting Bracket - In progress - Jaime
- PB13 - Improvements in Radar to allow more precises multi tracking - To Do - Eduardo, Djordje
