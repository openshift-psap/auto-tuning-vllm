#!/usr/bin/env bash
# Launch a clean, profile-driven Janus agentic tuning study.
#
# The controller output and credential-free invocation metadata are retained in
# /tmp/agentic-tuning-controller. Credential values are never written to either.
#
# For a study variation, copy this launcher or override TUNING_PROFILE,
# AGENT_REPORT_DIR, CONTROLLER_METADATA_DIR, or TMUX_SESSION_NAME.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
script_path="$script_dir/$(basename -- "${BASH_SOURCE[0]}")"

metadata_root=${CONTROLLER_LOG_ROOT:-/tmp/agentic-tuning-controller}
profile=${TUNING_PROFILE:-examples/agent/profiles/janus-nemotron-16k1k.yaml}
kubeconfig=${JANUS_KUBECONFIG:-/home/thibrahi/kubeconfigs/kubeconfig_files/janus}
anthropic_key_file=${ANTHROPIC_KEY_FILE:-/home/thibrahi/creds/claude-dev-key}
mlflow_credentials_file=${MLFLOW_CREDENTIALS_FILE:-/home/thibrahi/creds/mlflow}
report_dir=${AGENT_REPORT_DIR:-agent_reports/janus-nemotron-16k1k}
timestamp=${2:-$(date -u +%Y%m%dT%H%M%SZ)}
run_log_dir=${CONTROLLER_RUN_LOG_DIR:-"$metadata_root/$timestamp"}
metadata_dir=${CONTROLLER_METADATA_DIR:-"$run_log_dir"}
controller_log=${CONTROLLER_LOG_FILE:-"$run_log_dir/controller.log"}
session_name=${TMUX_SESSION_NAME:-janus-agentic-tuning-$timestamp}

mkdir -p "$run_log_dir"

read_credential() {
    local field=$1
    local value
    value=$(sed -n "s/^${field}[[:space:]]*[:=][[:space:]]*//p" "$mlflow_credentials_file")
    if [[ -z "$value" ]]; then
        echo "Missing $field in $mlflow_credentials_file" >&2
        exit 1
    fi
    printf '%s' "$value"
}

if [[ ${1:-} != "--foreground" ]]; then
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Controller session already exists: $session_name" >&2
        exit 1
    fi
    ln -sfn "$timestamp" "$metadata_root/latest"
    CONTROLLER_LOG_ROOT="$metadata_root" \
        CONTROLLER_RUN_LOG_DIR="$run_log_dir" \
        CONTROLLER_METADATA_DIR="$metadata_dir" \
        CONTROLLER_LOG_FILE="$controller_log" \
        tmux new-session -d -s "$session_name" \
        "exec $(printf '%q' "$script_path") --foreground $(printf '%q' "$timestamp")"
    echo "Controller session: $session_name"
    echo "Controller log: $controller_log"
    echo "Study logs: $run_log_dir"
    echo "Latest logs: $metadata_root/latest"
    exit 0
fi

mlflow_url=$(read_credential URL)
mlflow_user=$(read_credential User)
mlflow_pass=$(read_credential Pass)

cd "$repo_dir"
exec env \
    MLFLOW_TRACKING_URI="https://$mlflow_url" \
    MLFLOW_TRACKING_USERNAME="$mlflow_user" \
    MLFLOW_TRACKING_PASSWORD="$mlflow_pass" \
    MLFLOW_TRACKING_INSECURE_TLS=true \
    MLFLOW_WORKSPACE=thibrahi \
    .venv/bin/auto-tune-vllm agent \
    --tuning-profile "$profile" \
    --kubeconfig "$kubeconfig" \
    --api-key-file "$anthropic_key_file" \
    --claude-model claude-sonnet-4-6 \
    --no-vertex \
    --max-iterations 100 \
    --mlflow-uri "https://$mlflow_url" \
    --mlflow-experiment thibrahi \
    --mlflow-workspace thibrahi \
    --controller-metadata-dir "$metadata_dir" \
    --output "$report_dir-$timestamp" \
    >"$controller_log" 2>&1
