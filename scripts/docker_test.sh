#!/bin/bash
# Build the test image, start PostgreSQL 14 and run colcon build + test for b2_radar and rigsa_alpha in Docker.
# Same setup as .github/workflows/tests.yml. Usage: scripts/docker_test.sh [extra colcon test args]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE=rigsa-test:humble
NET=rigsa-test-net
PG=rigsa-test-pg

docker build -t "$IMAGE" -f "$REPO/docker/Dockerfile" "$REPO/docker"
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
docker rm -f "$PG" >/dev/null 2>&1 || true
trap 'docker rm -f "$PG" >/dev/null 2>&1 || true; docker network rm "$NET" >/dev/null 2>&1 || true' EXIT

docker run -d --name "$PG" --network "$NET" \
    -e POSTGRES_USER=rigsa -e POSTGRES_PASSWORD=rigsa -e POSTGRES_DB=rigsa \
    --health-cmd pg_isready --health-interval 2s --health-retries 15 postgres:14 >/dev/null
until [ "$(docker inspect -f '{{.State.Health.Status}}' "$PG")" = healthy ]; do sleep 1; done

# Repo is mounted read-only; build happens in a container-local copy so no build/install/log lands in the repo.
docker run --rm --network "$NET" -v "$REPO:/repo:ro" \
    -e RIGSA_TEST_DSN="host=$PG port=5432 dbname=rigsa user=rigsa password=rigsa" \
    "$IMAGE" bash -c '
        set -e
        mkdir -p /ws && cp -r /repo/src /repo/bags /ws/ && cd /ws
        source /opt/ros/humble/setup.bash
        colcon build --packages-select b2_radar rigsa_alpha
        source install/setup.bash
        colcon test --packages-select b2_radar rigsa_alpha --event-handlers console_direct+ '"$*"'
        colcon test-result --verbose'
