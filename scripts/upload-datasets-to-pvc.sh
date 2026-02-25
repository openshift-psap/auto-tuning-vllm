#!/bin/bash
# Upload MLPerf datasets to model PVC
# This is an alternative to baking datasets into the Docker image

NAMESPACE="autotune-aanya"
PVC_NAME="model-cache-pvc"
LOCAL_DATASET_DIR="/Users/aansharm/mlperf-build/mlperf-datasets"

echo "Uploading datasets from $LOCAL_DATASET_DIR to PVC $PVC_NAME..."

# Create uploader pod
kubectl run dataset-uploader --image=busybox -n $NAMESPACE \
  --overrides="{
    \"spec\": {
      \"volumes\": [{
        \"name\": \"models\",
        \"persistentVolumeClaim\": {\"claimName\": \"$PVC_NAME\"}
      }],
      \"containers\": [{
        \"name\": \"uploader\",
        \"image\": \"busybox\",
        \"command\": [\"sleep\", \"3600\"],
        \"volumeMounts\": [{
          \"name\": \"models\",
          \"mountPath\": \"/mnt/models\"
        }]
      }]
    }
  }"

# Wait for pod to be ready
echo "Waiting for uploader pod to be ready..."
kubectl wait --for=condition=Ready pod/dataset-uploader -n $NAMESPACE --timeout=60s

# Create datasets directory on PVC
echo "Creating datasets directory on PVC..."
kubectl exec -n $NAMESPACE dataset-uploader -- mkdir -p /mnt/models/datasets

# Copy datasets
echo "Copying datasets to PVC (this may take a few minutes for 584MB)..."
kubectl cp $LOCAL_DATASET_DIR/v4 $NAMESPACE/dataset-uploader:/mnt/models/datasets/v4

# Verify upload
echo ""
echo "Verifying upload..."
kubectl exec -n $NAMESPACE dataset-uploader -- ls -lah /mnt/models/datasets/v4/

# Cleanup
echo ""
echo "Cleaning up uploader pod..."
kubectl delete pod dataset-uploader -n $NAMESPACE

echo ""
echo "✅ Datasets uploaded successfully to PVC!"
echo ""
echo "Next steps:"
echo "1. Update study config dataset_path to: /mnt/models/datasets/v4/perf_eval_ref.parquet"
echo "2. Ensure model_pvc: 'model-cache-pvc' is set in k8s config"
echo "3. Rebuild Docker image without COPY mlperf-datasets commands (optional - reduces image size)"
