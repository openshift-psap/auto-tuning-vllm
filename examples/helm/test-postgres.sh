#!/bin/bash
# Script to test PostgreSQL deployment
# Usage: ./test-postgres.sh [postgres-service-name]

set -e

# Configuration
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
NAMESPACE="llm-d-trials"
POSTGRES_SERVICE="${1:-postgresql}"
TEST_POD_NAME="postgres-test-pod"
TEST_POD_FILE="$(dirname "$0")/postgres-test-pod.yaml"

# Export KUBECONFIG
export KUBECONFIG

echo "Testing PostgreSQL deployment..."
echo "KUBECONFIG: $KUBECONFIG"
echo "Namespace: $NAMESPACE"
echo "PostgreSQL Service: $POSTGRES_SERVICE"
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

# Check if PostgreSQL service exists, or try to auto-detect
echo "Checking PostgreSQL service..."
if ! kubectl get service "$POSTGRES_SERVICE" -n "$NAMESPACE" &> /dev/null; then
    echo "Service '$POSTGRES_SERVICE' not found. Attempting to auto-detect..."
    # Try to find any PostgreSQL service
    DETECTED_SERVICE=$(kubectl get services -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep -i postgres | head -1 || echo "")
    if [ -n "$DETECTED_SERVICE" ]; then
        echo "Found PostgreSQL service: $DETECTED_SERVICE"
        POSTGRES_SERVICE="$DETECTED_SERVICE"
    else
        echo "Error: PostgreSQL service not found in namespace '$NAMESPACE'"
        echo ""
        echo "Available services:"
        kubectl get services -n "$NAMESPACE"
        exit 1
    fi
fi
echo "✓ Using PostgreSQL service: $POSTGRES_SERVICE"
echo ""

# Check if PostgreSQL pod is ready
echo "Checking PostgreSQL pod status..."
# Try multiple label selectors
POSTGRES_POD=$(kubectl get pods -n "$NAMESPACE" -l app=postgresql -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [ -z "$POSTGRES_POD" ]; then
    POSTGRES_POD=$(kubectl get pods -n "$NAMESPACE" -l component=postgresql -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
fi
if [ -z "$POSTGRES_POD" ]; then
    echo "Warning: No PostgreSQL pod found with standard labels"
    echo "Trying to find any PostgreSQL pod..."
    POSTGRES_POD=$(kubectl get pods -n "$NAMESPACE" | grep -i postgres | head -1 | awk '{print $1}' || echo "")
fi

if [ -n "$POSTGRES_POD" ]; then
    echo "Found PostgreSQL pod: $POSTGRES_POD"
    if kubectl wait --for=condition=ready pod "$POSTGRES_POD" -n "$NAMESPACE" --timeout=30s &> /dev/null; then
        echo "✓ PostgreSQL pod is ready"
    else
        echo "⚠ PostgreSQL pod is not ready yet"
        echo "Pod status:"
        kubectl get pod "$POSTGRES_POD" -n "$NAMESPACE"
    fi
else
    echo "⚠ Could not find PostgreSQL pod"
fi
echo ""

# Clean up any existing test pod
echo "Cleaning up any existing test pod..."
kubectl delete pod "$TEST_POD_NAME" -n "$NAMESPACE" --ignore-not-found=true &> /dev/null
sleep 2
echo ""

# Update test pod with correct service name and secret
echo "Updating test pod configuration for service: $POSTGRES_SERVICE"
# Detect secret name
POSTGRES_SECRET=$(kubectl get secret -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep -i postgres | grep -i secret | head -1 || echo "postgresql-secret")
if [ -z "$POSTGRES_SECRET" ] || [ "$POSTGRES_SECRET" = "postgresql-secret" ]; then
    # Try to find secret from service labels
    POSTGRES_SECRET=$(kubectl get service "$POSTGRES_SERVICE" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.app\.kubernetes\.io/instance}' 2>/dev/null || echo "")
    if [ -n "$POSTGRES_SECRET" ]; then
        POSTGRES_SECRET="${POSTGRES_SECRET}-vllm-server-postgresql-secret"
    else
        POSTGRES_SECRET="postgresql-secret"
    fi
fi

# Create a temporary file with updated service name and secret
TEMP_FILE=$(mktemp)
sed -e "s|value: postgresql\.llm-d-trials\.svc\.cluster\.local|value: ${POSTGRES_SERVICE}.${NAMESPACE}.svc.cluster.local|g" \
    -e "s|name: postgresql-secret|name: ${POSTGRES_SECRET}|g" \
    "$TEST_POD_FILE" > "$TEMP_FILE"
TEST_POD_FILE="$TEMP_FILE"
trap "rm -f $TEMP_FILE" EXIT
echo "Using secret: $POSTGRES_SECRET"

# Deploy test pod
echo "Deploying test pod..."
kubectl apply -f "$TEST_POD_FILE"
echo ""

# Wait for test pod to be ready
echo "Waiting for test pod to be ready..."
if kubectl wait --for=condition=ready pod "$TEST_POD_NAME" -n "$NAMESPACE" --timeout=60s; then
    echo "✓ Test pod is ready"
else
    echo "Error: Test pod failed to become ready"
    kubectl describe pod "$TEST_POD_NAME" -n "$NAMESPACE"
    exit 1
fi
echo ""

# Run connectivity test
echo "Running PostgreSQL connectivity tests..."
echo "=========================================="
echo ""

# Test 1: Basic connectivity
echo "Test 1: Basic connectivity (pg_isready)..."
if kubectl exec "$TEST_POD_NAME" -n "$NAMESPACE" -- pg_isready -h "$POSTGRES_SERVICE.$NAMESPACE.svc.cluster.local" -p 5432 -U postgres; then
    echo "✓ PostgreSQL is accepting connections"
else
    echo "✗ Failed to connect to PostgreSQL"
    exit 1
fi
echo ""

# Test 2: Database connection
echo "Test 2: Database connection..."
if kubectl exec "$TEST_POD_NAME" -n "$NAMESPACE" -- psql -h "$POSTGRES_SERVICE.$NAMESPACE.svc.cluster.local" -U postgres -d postgres -c "SELECT version();" &> /dev/null; then
    echo "✓ Successfully connected to database"
    echo ""
    echo "PostgreSQL version:"
    kubectl exec "$TEST_POD_NAME" -n "$NAMESPACE" -- psql -h "$POSTGRES_SERVICE.$NAMESPACE.svc.cluster.local" -U postgres -d postgres -t -c "SELECT version();" | head -1
else
    echo "✗ Failed to connect to database"
    exit 1
fi
echo ""

# Test 3: Create and query a test table
echo "Test 3: Create and query test table..."
kubectl exec "$TEST_POD_NAME" -n "$NAMESPACE" -- psql -h "$POSTGRES_SERVICE.$NAMESPACE.svc.cluster.local" -U postgres -d postgres <<EOF
CREATE TABLE IF NOT EXISTS test_table (
    id SERIAL PRIMARY KEY,
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO test_table (message) VALUES ('PostgreSQL test successful!');
SELECT * FROM test_table;
DROP TABLE test_table;
EOF

if [ $? -eq 0 ]; then
    echo "✓ Test table operations successful"
else
    echo "✗ Test table operations failed"
    exit 1
fi
echo ""

# Test 4: List databases
echo "Test 4: List databases..."
echo "Available databases:"
kubectl exec "$TEST_POD_NAME" -n "$NAMESPACE" -- psql -h "$POSTGRES_SERVICE.$NAMESPACE.svc.cluster.local" -U postgres -d postgres -t -c "\l" | head -10
echo ""

echo "=========================================="
echo "✓ All PostgreSQL tests passed!"
echo ""
echo "Test pod '$TEST_POD_NAME' is still running for manual testing."
echo "You can connect to it with:"
echo "  kubectl exec -it $TEST_POD_NAME -n $NAMESPACE -- /bin/bash"
echo ""
echo "Inside the pod, you can use:"
echo "  psql -h $POSTGRES_SERVICE.$NAMESPACE.svc.cluster.local -U postgres -d postgres"
echo ""
echo "To clean up the test pod:"
echo "  kubectl delete pod $TEST_POD_NAME -n $NAMESPACE"
