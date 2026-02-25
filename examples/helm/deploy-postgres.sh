#!/bin/bash
# Script to deploy PostgreSQL StatefulSet to Kubernetes cluster
# Usage: ./deploy-postgres.sh [release-name]

set -e

# Configuration
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
NAMESPACE="llm-d-trials"
RELEASE_NAME="${1:-postgresql}"
CHART_PATH="$(dirname "$0")"

# Export KUBECONFIG
export KUBECONFIG

echo "Deploying PostgreSQL StatefulSet..."
echo "KUBECONFIG: $KUBECONFIG"
echo "Namespace: $NAMESPACE"
echo "Release Name: $RELEASE_NAME"
echo "Chart Path: $CHART_PATH"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check if helm is available
if ! command -v helm &> /dev/null; then
    echo "Error: helm is not installed or not in PATH"
    exit 1
fi

# Verify cluster access
echo "Verifying cluster access..."
if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Cannot access cluster. Please check your KUBECONFIG"
    exit 1
fi
echo "✓ Cluster access verified"
echo ""

# Create namespace if it doesn't exist
echo "Ensuring namespace '$NAMESPACE' exists..."
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "Creating namespace '$NAMESPACE'..."
    kubectl create namespace "$NAMESPACE"
    echo "✓ Namespace created"
else
    echo "✓ Namespace already exists"
fi
echo ""

# Deploy PostgreSQL using Helm
echo "Deploying PostgreSQL StatefulSet with Helm..."
helm upgrade --install "$RELEASE_NAME" "$CHART_PATH" \
    --namespace "$NAMESPACE" \
    --set postgresql.enabled=true \
    --set postgresql.namespace="$NAMESPACE" \
    --set postgresql.persistence.storageClass=nfs-storage \
    --wait \
    --timeout 10m

echo ""
echo "✓ PostgreSQL StatefulSet deployed successfully!"
echo ""
echo "To check the status:"
echo "  kubectl get statefulset -n $NAMESPACE"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl get pvc -n $NAMESPACE"
echo ""
echo "To get PostgreSQL connection details:"
echo "  kubectl get secret ${RELEASE_NAME}-postgresql-secret -n $NAMESPACE -o jsonpath='{.data.postgres-user}' | base64 -d"
echo "  kubectl get secret ${RELEASE_NAME}-postgresql-secret -n $NAMESPACE -o jsonpath='{.data.postgres-password}' | base64 -d"
echo ""
echo "Service endpoint:"
echo "  ${RELEASE_NAME}-postgresql.${NAMESPACE}.svc.cluster.local:5432"
