#!/usr/bin/env bash
set -eu

if [ $# -ne 2 ]; then
    echo "Usage: $0 <dockerfile> <name>"
    echo "  e.g. $0 dockerfiles/visualise-sae.dockerfile sae-visualise"
    exit 1
fi

DOCKERFILE="$1"
TAG="registry.example.com/sae-01/$2:latest"

docker build -f "$DOCKERFILE" -t "$TAG" .
docker push "$TAG"
