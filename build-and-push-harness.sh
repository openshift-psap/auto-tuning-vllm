#!/bin/bash
# Script to build and push the updated MLPerf harness image

set -e

echo "=========================================="
echo "Building Updated MLPerf Harness Image"
echo "=========================================="

# Configuration
IMAGE_NAME="quay.io/rh-ee-nmiriyal/mlperf-6.0"
IMAGE_TAG="harness-updated"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "Image will be tagged as: ${FULL_IMAGE}"
echo ""

# Build the image
echo "Step 1: Building Docker image..."
docker build -f Dockerfile.harness-updated -t ${FULL_IMAGE} .

echo ""
echo "✅ Image built successfully!"
echo ""

# Ask for confirmation before pushing
read -p "Do you want to push to ${FULL_IMAGE}? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Step 2: Logging in to quay.io..."
    docker login quay.io

    echo ""
    echo "Step 3: Pushing image..."
    docker push ${FULL_IMAGE}

    echo ""
    echo "=========================================="
    echo "✅ SUCCESS!"
    echo "=========================================="
    echo ""
    echo "Image pushed to: ${FULL_IMAGE}"
    echo ""
    echo "Next steps:"
    echo "1. Update your study_config.yaml to use: ${FULL_IMAGE}"
    echo "2. Run your optimization again"
else
    echo ""
    echo "Push cancelled. Image is built locally as: ${FULL_IMAGE}"
    echo ""
    echo "To push later, run:"
    echo "  docker login quay.io"
    echo "  docker push ${FULL_IMAGE}"
fi
