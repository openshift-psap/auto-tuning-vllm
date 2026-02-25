#!/bin/bash
# Script to deploy PostgreSQL StatefulSet using standalone YAML
# Usage: ./deploy-postgres-standalone.sh

set -e

# Configuration
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
NAMESPACE="llm-d-trials"
MANIFEST_FILE="$(dirname "$0")/postgres-standalone.yaml"

# Export KUBECONFIG
export KUBECONFIG

echo "Deploying PostgreSQL StatefulSet (standalone)..."
echo "KUBECONFIG: $KUBECONFIG"
echo "Namespace: $NAMESPACE"
echo "Manifest: $MANIFEST_FILE"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
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

# Deploy PostgreSQL
echo "Deploying PostgreSQL StatefulSet..."
kubectl apply -f "$MANIFEST_FILE"

echo ""
echo "Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=ready pod -l app=postgresql -n "$NAMESPACE" --timeout=300s

echo ""
echo "✓ PostgreSQL StatefulSet deployed successfully!"
echo ""
echo "To check the status:"
echo "  kubectl get statefulset -n $NAMESPACE"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl get pvc -n $NAMESPACE"
echo ""
echo "To get PostgreSQL connection details:"
echo "  kubectl get secret postgresql-secret -n $NAMESPACE -o jsonpath='{.data.postgres-user}' | base64 -d && echo"
echo "  kubectl get secret postgresql-secret -n $NAMESPACE -o jsonpath='{.data.postgres-password}' | base64 -d && echo"
echo ""
echo "Service endpoint:"
echo "  postgresql.${NAMESPACE}.svc.cluster.local:5432"
