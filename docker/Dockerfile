# Test image mirroring .github/workflows/tests.yml (ROS 2 Humble on Ubuntu 22.04).
# Only dependencies live here; the repo is mounted at run time (see scripts/docker_test.sh).
FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pytest python3-numpy python3-scipy python3-psycopg2 \
        python3-serial ros-humble-rosbag2-py ros-humble-rosbag2-storage-default-plugins \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
