"""Helm utilities for vLLM deployment and Kubernetes Job management."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..core.trial import TrialConfig

logger = logging.getLogger(__name__)


def sanitize_release_name(name: str) -> str:
    """Sanitize name for Helm release (lowercase, alphanumeric and hyphens only).

    Args:
        name: Original name

    Returns:
        Sanitized name suitable for Helm release
    """
    # Convert to lowercase
    name = name.lower()
    # Replace underscores and spaces with hyphens
    name = name.replace("_", "-").replace(" ", "-")
    # Remove invalid characters (keep only alphanumeric and hyphens)
    name = re.sub(r"[^a-z0-9-]", "", name)
    # Remove consecutive hyphens
    name = re.sub(r"-+", "-", name)
    # Remove leading/trailing hyphens
    name = name.strip("-")
    # Ensure it starts with a letter or number
    if name and not name[0].isalnum():
        name = "release-" + name
    # Helm release names must be <= 53 characters
    if len(name) > 53:
        name = name[:53].rstrip("-")
    return name or "release"


def convert_vllm_args_to_helm_values(vllm_args: List[str]) -> Dict[str, Any]:
    """Convert vLLM CLI arguments to Helm values format.

    Args:
        vllm_args: List of vLLM CLI arguments (e.g., ["--max-num-seqs", "256"])

    Returns:
        Dictionary of Helm values
    """
    values = {}
    i = 0
    while i < len(vllm_args):
        arg = vllm_args[i]
        if arg.startswith("--"):
            # Remove -- prefix
            key = arg[2:]
            # Convert to nested dict format if needed (e.g., max-num-seqs -> maxNumSeqs)
            # For now, keep as-is but convert dashes to camelCase for nested keys
            key_parts = key.split("-")
            if len(key_parts) > 1:
                # Convert to camelCase
                camel_key = key_parts[0] + "".join(
                    p.capitalize() for p in key_parts[1:]
                )
            else:
                camel_key = key

            # Check if next arg is a value (not another flag)
            if i + 1 < len(vllm_args) and not vllm_args[i + 1].startswith("--"):
                value = vllm_args[i + 1]
                # Try to convert to appropriate type
                try:
                    if value.lower() in ("true", "false"):
                        values[camel_key] = value.lower() == "true"
                    elif "." in value:
                        values[camel_key] = float(value)
                    else:
                        values[camel_key] = int(value)
                except ValueError:
                    values[camel_key] = value
                i += 2
            else:
                # Boolean flag (present = true)
                values[camel_key] = True
                i += 1
        else:
            i += 1

    return values


def generate_gaie_values(
    trial_config: TrialConfig, helm_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate GAIE Helm values from TrialConfig.

    Args:
        trial_config: Trial configuration
        helm_config: Helm-specific configuration

    Returns:
        Dictionary of GAIE Helm values ready for deployment
    """
    from pathlib import Path

    # Start with base GAIE values
    gaie_values = {
        "inferencePool": {
            "targetPorts": [{"number": 8000}],
            "modelServerType": "vllm",
            "modelServers": {
                "matchLabels": {
                    "llm-d.ai/role": "decode",
                    "llm-d.ai/inferenceServing": "true",
                }
            },
        },
        "inferenceExtension": {
            "replicas": 1,
            "image": {
                "name": "llm-d-inference-scheduler",
                "hub": "ghcr.io/llm-d",
                "tag": "v0.4.0",
                "pullPolicy": "Always",
            },
            "extProcPort": 9002,
            "extraContainerPorts": [
                {"name": "zmq", "containerPort": 5557, "protocol": "TCP"}
            ],
            "extraServicePorts": [
                {"name": "zmq", "port": 5557, "targetPort": 5557, "protocol": "TCP"}
            ],
            "flags": {
                "kv-cache-usage-percentage-metric": "vllm:kv_cache_usage_perc",
                "v": 4,
            },
            "pluginsConfigFile": "precise-prefix-cache-config.yaml",
            "pluginsCustomConfig": {
                "precise-prefix-cache-config.yaml": """apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: single-profile-handler
  - type: precise-prefix-cache-scorer
    parameters:
      indexerConfig:
        tokenProcessorConfig:
          blockSize: 64
          hashSeed: "42"
        tokenizersPoolConfig:
          hf:
            tokenizersCacheDir: "/tmp/tokenizers"
  - type: kv-cache-utilization-scorer
  - type: queue-scorer
  - type: max-score-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: precise-prefix-cache-scorer
        weight: 3.0
      - pluginRef: kv-cache-utilization-scorer
        weight: 2.0
      - pluginRef: queue-scorer
        weight: 2.0
      - pluginRef: max-score-picker"""
            },
        },
        "provider": {"name": "istio"},
    }

    # Load GAIE values template if provided and merge (template takes precedence)
    if helm_config.get("gaie_values_template"):
        gaie_values_path = Path(helm_config["gaie_values_template"])
        if gaie_values_path.exists():
            with open(gaie_values_path) as f:
                template_values = yaml.safe_load(f) or {}

                # Deep merge template values into gaie_values
                def deep_merge(base, update):
                    for key, value in update.items():
                        if (
                            key in base
                            and isinstance(base[key], dict)
                            and isinstance(value, dict)
                        ):
                            deep_merge(base[key], value)
                        else:
                            base[key] = value

                deep_merge(gaie_values, template_values)

    # Apply trial-specific GAIE parameters
    # Common tunable parameters:
    # - gaie_replicas: inferenceExtension.replicas
    # - gaie_precise_prefix_cache_scorer_weight: pluginsCustomConfig schedulingProfiles[0].plugins[0].weight
    # - gaie_kv_cache_utilization_scorer_weight: pluginsCustomConfig schedulingProfiles[0].plugins[1].weight
    # - gaie_queue_scorer_weight: pluginsCustomConfig schedulingProfiles[0].plugins[2].weight
    # - gaie_block_size: pluginsCustomConfig tokenProcessorConfig.blockSize
    # - gaie_hash_seed: pluginsCustomConfig tokenProcessorConfig.hashSeed

    if trial_config.gaie_parameters:
        # Update replicas if specified
        if "gaie_replicas" in trial_config.gaie_parameters:
            gaie_values["inferenceExtension"]["replicas"] = int(
                trial_config.gaie_parameters["gaie_replicas"]
            )

        # Update plugin weights if specified
        if any("weight" in key for key in trial_config.gaie_parameters.keys()):
            # Parse the plugins config YAML to update weights
            plugins_config_str = gaie_values["inferenceExtension"][
                "pluginsCustomConfig"
            ]["precise-prefix-cache-config.yaml"]
            plugins_config = yaml.safe_load(plugins_config_str)

            if (
                "schedulingProfiles" in plugins_config
                and len(plugins_config["schedulingProfiles"]) > 0
            ):
                profile = plugins_config["schedulingProfiles"][0]
                if "plugins" in profile:
                    for plugin in profile["plugins"]:
                        plugin_ref = plugin.get("pluginRef", "")
                        if (
                            "gaie_precise_prefix_cache_scorer_weight"
                            in trial_config.gaie_parameters
                            and plugin_ref == "precise-prefix-cache-scorer"
                        ):
                            plugin["weight"] = float(
                                trial_config.gaie_parameters[
                                    "gaie_precise_prefix_cache_scorer_weight"
                                ]
                            )
                        elif (
                            "gaie_kv_cache_utilization_scorer_weight"
                            in trial_config.gaie_parameters
                            and plugin_ref == "kv-cache-utilization-scorer"
                        ):
                            plugin["weight"] = float(
                                trial_config.gaie_parameters[
                                    "gaie_kv_cache_utilization_scorer_weight"
                                ]
                            )
                        elif (
                            "gaie_queue_scorer_weight" in trial_config.gaie_parameters
                            and plugin_ref == "queue-scorer"
                        ):
                            plugin["weight"] = float(
                                trial_config.gaie_parameters["gaie_queue_scorer_weight"]
                            )

            # Update the plugins config string
            gaie_values["inferenceExtension"]["pluginsCustomConfig"][
                "precise-prefix-cache-config.yaml"
            ] = yaml.dump(plugins_config, default_flow_style=False, allow_unicode=True)

        # Update block size and hash seed if specified
        if (
            "gaie_block_size" in trial_config.gaie_parameters
            or "gaie_hash_seed" in trial_config.gaie_parameters
        ):
            plugins_config_str = gaie_values["inferenceExtension"][
                "pluginsCustomConfig"
            ]["precise-prefix-cache-config.yaml"]
            plugins_config = yaml.safe_load(plugins_config_str)

            # Find the precise-prefix-cache-scorer plugin
            for plugin in plugins_config.get("plugins", []):
                if plugin.get("type") == "precise-prefix-cache-scorer":
                    params = plugin.get("parameters", {})
                    indexer_config = params.get("indexerConfig", {})
                    token_processor_config = indexer_config.get(
                        "tokenProcessorConfig", {}
                    )

                    if "gaie_block_size" in trial_config.gaie_parameters:
                        token_processor_config["blockSize"] = int(
                            trial_config.gaie_parameters["gaie_block_size"]
                        )
                    if "gaie_hash_seed" in trial_config.gaie_parameters:
                        token_processor_config["hashSeed"] = str(
                            trial_config.gaie_parameters["gaie_hash_seed"]
                        )

                    break

            # Update the plugins config string
            gaie_values["inferenceExtension"]["pluginsCustomConfig"][
                "precise-prefix-cache-config.yaml"
            ] = yaml.dump(plugins_config, default_flow_style=False, allow_unicode=True)

    return gaie_values


def generate_helm_values(
    trial_config: TrialConfig, helm_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate Helm values from TrialConfig.

    Args:
        trial_config: Trial configuration
        helm_config: Helm-specific configuration (chart path, namespace, etc.)

    Returns:
        Dictionary of Helm values ready for deployment
    """
    values = {}

    # Start with template if provided
    if helm_config.get("values_template"):
        template_path = Path(helm_config["values_template"])
        if template_path.exists():
            with open(template_path) as f:
                values = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Helm values template not found: {template_path}")

    # Require explicit chart type specification
    chart_type = helm_config.get("chart_type")
    if not chart_type:
        raise ValueError(
            "chart_type must be explicitly specified in Helm configuration. "
            "Set chart_type to either 'llm-d-modelservice' or 'vllm'. "
            "Example: helm: { chart_type: 'llm-d-modelservice', ... }"
        )

    chart_type_lower = chart_type.lower()
    if chart_type_lower not in ("llm-d-modelservice", "vllm"):
        raise ValueError(
            f"Invalid chart_type: '{chart_type}'. "
            "Must be either 'llm-d-modelservice' or 'vllm'."
        )

    is_llm_d_modelservice = chart_type_lower == "llm-d-modelservice"

    # Convert vLLM args to Helm values
    vllm_values = convert_vllm_args_to_helm_values(trial_config.vllm_args)

    # Merge vLLM values into main values dict
    if "decode" not in values:
        values["decode"] = {}
    if "containers" not in values["decode"]:
        values["decode"]["containers"] = []
    if not values["decode"]["containers"]:
        values["decode"]["containers"].append({})

    container = values["decode"]["containers"][0]

    # Set container image if not present (required by llm-d-modelservice chart)
    # For prefix-aware routing experiments, use llm-d-cuda image
    # For disaggregation experiments, use llm-d-inference-sim image
    if "image" not in container:
        if is_llm_d_modelservice:
            # Default to llm-d-cuda image for prefix-aware routing experiments
            container["image"] = "ghcr.io/llm-d/llm-d-cuda:v0.3.1"
        else:
            # For vanilla vLLM charts, use vLLM image
            container["image"] = "ghcr.io/vllm-project/vllm-openai:latest"

    # Set modelCommand if not present
    if "modelCommand" not in container:
        container["modelCommand"] = "vllmServe" if is_llm_d_modelservice else None

    # Check if using custom modelCommand (for prefix-aware routing with KV events)
    is_custom_command = container.get("modelCommand") == "custom"

    if is_llm_d_modelservice:
        # For llm-d-modelservice: Chart constructs args from parallelism, modelArtifacts, etc.
        # Only add non-standard vLLM args to container.args (they get appended)
        # Standard args like --model, --tensor-parallel-size are constructed by chart

        # Initialize args list for additional parameters
        if "args" not in container:
            container["args"] = []

        # Map parallelism parameters from trial config
        # tensor_parallel_size -> decode.parallelism.tensor
        if "tensor_parallel_size" in trial_config.parameters:
            if "parallelism" not in values["decode"]:
                values["decode"]["parallelism"] = {}
            values["decode"]["parallelism"]["tensor"] = trial_config.parameters[
                "tensor_parallel_size"
            ]

        # data_parallel_size -> decode.parallelism.data
        if "data_parallel_size" in trial_config.parameters:
            if "parallelism" not in values["decode"]:
                values["decode"]["parallelism"] = {}
            values["decode"]["parallelism"]["data"] = trial_config.parameters[
                "data_parallel_size"
            ]

        # data_parallel_size_local -> decode.parallelism.dataLocal
        if "data_parallel_size_local" in trial_config.parameters:
            if "parallelism" not in values["decode"]:
                values["decode"]["parallelism"] = {}
            values["decode"]["parallelism"]["dataLocal"] = trial_config.parameters[
                "data_parallel_size_local"
            ]

        if is_custom_command:
            # For custom command: construct shell command string
            # Format: "vllm serve MODEL --host 0.0.0.0 --port 8200 [args...]"
            cmd_parts = ["vllm", "serve"]

            # Add model
            if trial_config.benchmark_config and trial_config.benchmark_config.model:
                cmd_parts.append(trial_config.benchmark_config.model)

            # Add standard args
            cmd_parts.extend(["--host", "0.0.0.0", "--port", "8200"])

            # Add all vLLM args (including standard ones for custom command)
            for arg in trial_config.vllm_args:
                if arg not in cmd_parts:
                    cmd_parts.append(arg)

            # Join into single shell command string
            container["args"] = [" \\\n        ".join(cmd_parts)]
        else:
            # For vllmServe/imageDefault: Add only non-standard vLLM args to container.args
            # Standard args (--model, --tensor-parallel-size, --data-parallel-size, etc.) are constructed by chart
            standard_args = {
                "--model",
                "--tensor-parallel-size",
                "--data-parallel-size",
                "--data-parallel-size-local",
                "--served-model-name",
                "--port",
                "--host",
            }

            for arg in trial_config.vllm_args:
                # Skip standard args that chart will construct
                if arg in standard_args or any(
                    arg.startswith(f"{std}=") for std in standard_args
                ):
                    continue
                # Add non-standard args (e.g., --max-num-seqs, --gpu-memory-utilization)
                if arg not in container["args"]:
                    container["args"].append(arg)
    else:
        # For vanilla vLLM charts: Add all vLLM args directly to container.args
        if "args" not in container:
            container["args"] = []

        # Add all vLLM args
        for arg in trial_config.vllm_args:
            if arg not in container["args"]:
                container["args"].append(arg)

        # Ensure model is specified in args if not already present
        if trial_config.benchmark_config and trial_config.benchmark_config.model:
            has_model = any(
                arg.startswith("--model") or arg == trial_config.benchmark_config.model
                for arg in container["args"]
            )
            if not has_model:
                container["args"].extend(
                    ["--model", trial_config.benchmark_config.model]
                )

    # Add environment variables
    if "env" not in container:
        container["env"] = []

    env_dict = {
        env["name"]: env.get("value", "") for env in container["env"] if "name" in env
    }
    env_dict.update(trial_config.environment_vars)

    # Set HF_HOME to PVC mount path if model_pvc is specified
    model_pvc = helm_config.get("model_pvc")
    model_mount_path = "/mnt/models"
    if model_pvc:
        env_dict["HF_HOME"] = model_mount_path

    container["env"] = [{"name": k, "value": str(v)} for k, v in env_dict.items()]

    # Set model from benchmark config
    if "modelArtifacts" not in values:
        values["modelArtifacts"] = {}
    if trial_config.benchmark_config and trial_config.benchmark_config.model:
        values["modelArtifacts"]["uri"] = f"hf://{trial_config.benchmark_config.model}"
        values["modelArtifacts"]["name"] = trial_config.benchmark_config.model

    # Set resource requirements
    if "resources" not in container:
        container["resources"] = {}
    if "limits" not in container["resources"]:
        container["resources"]["limits"] = {}
    if "requests" not in container["resources"]:
        container["resources"]["requests"] = {}

    num_gpus = trial_config.resource_requirements.get("num_gpus", 1)
    container["resources"]["limits"]["nvidia.com/gpu"] = str(int(num_gpus))
    container["resources"]["requests"]["nvidia.com/gpu"] = str(int(num_gpus))

    # Add PVC volume mount if model_pvc is specified
    model_pvc = helm_config.get("model_pvc")
    model_mount_path = "/mnt/models"
    if model_pvc:
        # Add volumeMount to container
        if "volumeMounts" not in container:
            container["volumeMounts"] = []
        # Check if volumeMount already exists
        if not any(vm.get("name") == "model-pvc" for vm in container["volumeMounts"]):
            container["volumeMounts"].append(
                {
                    "name": "model-pvc",
                    "mountPath": model_mount_path,
                }
            )

        # Add volume to decode.volumes
        if "volumes" not in values.get("decode", {}):
            if "decode" not in values:
                values["decode"] = {}
            values["decode"]["volumes"] = []
        # Check if volume already exists
        if not any(v.get("name") == "model-pvc" for v in values["decode"]["volumes"]):
            values["decode"]["volumes"].append(
                {
                    "name": "model-pvc",
                    "persistentVolumeClaim": {
                        "claimName": model_pvc,
                    },
                }
            )

    # Ensure routing and service configuration for llm-d-modelservice chart
    # The chart creates services through routing.proxy sidecar configuration
    if is_llm_d_modelservice:
        # Configure routing section (required for service creation)
        if "routing" not in values:
            values["routing"] = {}
        # Set servicePort for routing (this is what the service will expose)
        if "servicePort" not in values["routing"]:
            values["routing"]["servicePort"] = 8000

        # Configure routing proxy (sidecar that creates the service)
        if "proxy" not in values["routing"]:
            values["routing"]["proxy"] = {}
        # Enable proxy if not explicitly disabled
        if "enabled" not in values["routing"]["proxy"]:
            values["routing"]["proxy"]["enabled"] = True
        # Set targetPort to match vLLM container port (8200 for prefix-aware routing)
        if "targetPort" not in values["routing"]["proxy"]:
            container_port = 8200
            if container.get("ports") and len(container["ports"]) > 0:
                # Find the http port
                for port in container["ports"]:
                    if port.get("name") == "http" or port.get("containerPort") == 8200:
                        container_port = port.get("containerPort", 8200)
                        break
            values["routing"]["proxy"]["targetPort"] = container_port
        # Set proxy image if not present
        if "image" not in values["routing"]["proxy"]:
            values["routing"]["proxy"]["image"] = (
                "ghcr.io/llm-d/llm-d-routing-sidecar:v0.4.0-rc.1"
            )

    return values


def get_service_url(release_name: str, namespace: str, helm_config: dict = None) -> str:
    """Get service URL from Helm release.

    Args:
        release_name: Helm release name (modelservice release)
        namespace: Kubernetes namespace
        helm_config: Optional Helm configuration to check for full stack deployment

    Returns:
        Service URL (e.g., "http://service-name.namespace.svc.cluster.local:8000/v1")
    """

    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.error("kubernetes library not available")
        raise RuntimeError("kubernetes library required for Helm backend")

    # DEBUG DISABLED: #region agent log
    # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
    # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"A","location":"helm_utils.py:334","message":"Starting get_service_url","data":{"release_name":release_name,"namespace":namespace,"helm_config":str(helm_config)},"timestamp":int(time.time()*1000)})+"\n")
    # DEBUG DISABLED: #endregion

    # For full stack deployment, use inference gateway service (LoadBalancer/NodePort)
    if helm_config and helm_config.get("deploy_full_stack", False):
        # DEBUG DISABLED: #region agent log
        # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
        # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"GAIE","location":"helm_utils.py:340","message":"Checking for full stack deployment","data":{"deploy_full_stack":True},"timestamp":int(time.time()*1000)})+"\n")
        # DEBUG DISABLED: #endregion
        release_name_postfix = helm_config.get("release_name_postfix", "kv-events")

        try:
            # DEBUG DISABLED: #region agent log
            # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
            # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"GAIE","location":"helm_utils.py:344","message":"Loading kubeconfig for gateway service","data":{},"timestamp":int(time.time()*1000)})+"\n")
            # DEBUG DISABLED: #endregion
            config.load_incluster_config()
        except config.ConfigException:
            # DEBUG DISABLED: #region agent log
            # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
            # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"GAIE","location":"helm_utils.py:348","message":"Loading kubeconfig from file","data":{},"timestamp":int(time.time()*1000)})+"\n")
            # DEBUG DISABLED: #endregion
            config.load_kube_config()

        v1 = client.CoreV1Api()

        # Try multiple gateway service name patterns (as per llm-d docs)
        # Pattern 1: infra-{postfix}-inference-gateway-istio (Istio gateway)
        # Pattern 2: infra-{postfix}-inference-gateway (regular gateway)
        gateway_service_names = [
            f"infra-{release_name_postfix}-inference-gateway-istio",
            f"infra-{release_name_postfix}-inference-gateway",
        ]

        service = None
        gateway_service_name = None
        for name in gateway_service_names:
            try:
                # DEBUG DISABLED: #region agent log
                # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"GAIE","location":"helm_utils.py:371","message":"Trying gateway service name","data":{"service_name":name,"namespace":namespace},"timestamp":int(time.time()*1000)})+"\n")
                # DEBUG DISABLED: #endregion
                service = v1.read_namespaced_service(name=name, namespace=namespace)
                gateway_service_name = name
                # DEBUG DISABLED: #region agent log
                # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"GAIE","location":"helm_utils.py:378","message":"Found gateway service","data":{"service_name":gateway_service_name},"timestamp":int(time.time()*1000)})+"\n")
                # DEBUG DISABLED: #endregion
                break
            except ApiException as e:
                if e.status == 404:
                    # Try next pattern
                    continue
                # Other errors should be raised
                raise

        # If not found by known patterns, search all services for pattern matching .*-inference-gateway(-.*)?$
        # This matches the approach in llm-d docs: select(.metadata.name | test(".*-inference-gateway(-.*)?$"))
        if service is None:
            # DEBUG DISABLED: #region agent log
            # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
            # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"GAIE","location":"helm_utils.py:390","message":"Searching all services for inference-gateway pattern","data":{"namespace":namespace},"timestamp":int(time.time()*1000)})+"\n")
            # DEBUG DISABLED: #endregion
            import re

            services = v1.list_namespaced_service(namespace=namespace)
            gateway_pattern = re.compile(r".*-inference-gateway(-.*)?$")
            for svc in services.items:
                if gateway_pattern.match(svc.metadata.name):
                    service = svc
                    gateway_service_name = svc.metadata.name
                    # DEBUG DISABLED: #region agent log
                    # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                    # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"GAIE","location":"helm_utils.py:400","message":"Found gateway service by pattern match","data":{"service_name":gateway_service_name},"timestamp":int(time.time()*1000)})+"\n")
                    # DEBUG DISABLED: #endregion
                    break

        if service:
            port = 80  # Default HTTP port for gateway
            if service.spec.ports:
                # Find HTTP port (usually 80)
                for svc_port in service.spec.ports:
                    if svc_port.name == "default" or svc_port.port == 80:
                        port = svc_port.port
                        break

            # Use LoadBalancer external IP if available, otherwise use ClusterIP
            if (
                service.spec.type == "LoadBalancer"
                and service.status.load_balancer.ingress
            ):
                ingress = service.status.load_balancer.ingress[0]
                host = ingress.hostname or ingress.ip
                # DEBUG DISABLED: #region agent log
                # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"GAIE","location":"helm_utils.py:365","message":"Using gateway LoadBalancer","data":{"service_name":gateway_service_name,"host":host,"port":port},"timestamp":int(time.time()*1000)})+"\n")
                # DEBUG DISABLED: #endregion
                return f"http://{host}:{port}/v1"
            elif service.spec.type == "NodePort" and service.spec.ports:
                node_port = service.spec.ports[0].node_port
                # For NodePort, we'd need node IP - use service DNS instead
                # DEBUG DISABLED: #region agent log
                # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"GAIE","location":"helm_utils.py:372","message":"Using gateway NodePort","data":{"service_name":gateway_service_name,"node_port":node_port,"port":port},"timestamp":int(time.time()*1000)})+"\n")
                # DEBUG DISABLED: #endregion
                return f"http://{gateway_service_name}.{namespace}.svc.cluster.local:{port}/v1"
            else:
                # ClusterIP - use ClusterIP directly
                cluster_ip = service.spec.cluster_ip
                if cluster_ip:
                    # DEBUG DISABLED: #region agent log
                    # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                    # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"GAIE","location":"helm_utils.py:380","message":"Using gateway ClusterIP","data":{"service_name":gateway_service_name,"cluster_ip":cluster_ip,"port":port},"timestamp":int(time.time()*1000)})+"\n")
                    # DEBUG DISABLED: #endregion
                    return f"http://{cluster_ip}:{port}/v1"
                return f"http://{gateway_service_name}.{namespace}.svc.cluster.local:{port}/v1"
        else:
            # No gateway service found with any pattern
            logger.warning(
                f"Gateway service not found in namespace '{namespace}' for patterns: {gateway_service_names}. "
                f"Falling back to modelservice service."
            )

    try:
        # Load kubeconfig
        # DEBUG DISABLED: #region agent log
        # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
        # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"A","location":"helm_utils.py:375","message":"Loading kubeconfig for modelservice","data":{},"timestamp":int(time.time()*1000)})+"\n")
        # DEBUG DISABLED: #endregion
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        v1 = client.CoreV1Api()

        # Try to find service created by Helm release
        # Helm typically creates services with release name or app name
        # DEBUG DISABLED: #region agent log
        # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
        # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"A","location":"helm_utils.py:386","message":"Listing services in namespace","data":{"namespace":namespace},"timestamp":int(time.time()*1000)})+"\n")
        # DEBUG DISABLED: #endregion
        services = v1.list_namespaced_service(namespace=namespace)

        # DEBUG DISABLED: #region agent log
        # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
        # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"A","location":"helm_utils.py:310","message":"Listed services","data":{"service_count":len(services.items),"service_names":[s.metadata.name for s in services.items]},"timestamp":int(time.time()*1000)})+"\n")
        # DEBUG DISABLED: #endregion

        for service in services.items:
            # Check if service belongs to this release
            labels = service.metadata.labels or {}
            if (
                labels.get("app.kubernetes.io/instance") == release_name
                or labels.get("release") == release_name
                or service.metadata.name.startswith(release_name)
            ):
                # Get service port (default to 8000 for vLLM)
                port = 8000
                if service.spec.ports:
                    port = service.spec.ports[0].port

                # DEBUG DISABLED: #region agent log
                # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"A","location":"helm_utils.py:320","message":"Found matching service","data":{"service_name":service.metadata.name,"port":port},"timestamp":int(time.time()*1000)})+"\n")
                # DEBUG DISABLED: #endregion

                # Construct URL based on service type
                if (
                    service.spec.type == "LoadBalancer"
                    and service.status.load_balancer.ingress
                ):
                    ingress = service.status.load_balancer.ingress[0]
                    host = ingress.hostname or ingress.ip
                    return f"http://{host}:{port}/v1"
                elif service.spec.type == "NodePort" and service.spec.ports:
                    node_port = service.spec.ports[0].node_port
                    # For NodePort, we'd need node IP - use service DNS instead
                    return f"http://{service.metadata.name}.{namespace}.svc.cluster.local:{port}/v1"
                else:
                    # ClusterIP (default)
                    # Use ClusterIP directly instead of DNS name for external access
                    cluster_ip = service.spec.cluster_ip
                    if cluster_ip:
                        return f"http://{cluster_ip}:{port}/v1"
                    # Fallback to DNS name if no ClusterIP (shouldn't happen)
                    return f"http://{service.metadata.name}.{namespace}.svc.cluster.local:{port}/v1"

        # Fallback: try to find service created by auto-tune-vllm (shorter name)
        # Check for service with auto-tune-vllm managed-by label
        for service in services.items:
            labels = service.metadata.labels or {}
            if labels.get("app.kubernetes.io/managed-by") == "auto-tune-vllm":
                port = 8000
                if service.spec.ports:
                    port = service.spec.ports[0].port
                # Use ClusterIP directly instead of DNS name for external access
                cluster_ip = service.spec.cluster_ip
                # DEBUG DISABLED: #region agent log
                # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"C","location":"helm_utils.py:352","message":"Found auto-tune-vllm service","data":{"service_name":service.metadata.name,"cluster_ip":cluster_ip,"port":port},"timestamp":int(time.time()*1000)})+"\n")
                # DEBUG DISABLED: #endregion
                if cluster_ip:
                    return f"http://{cluster_ip}:{port}/v1"
                # Fallback to DNS name if no ClusterIP
                return f"http://{service.metadata.name}.{namespace}.svc.cluster.local:{port}/v1"

        # Last resort: construct expected service name (may not exist and may be too long)
        service_name = f"{release_name}-llm-d-modelservice-decode"
        # DEBUG DISABLED: #region agent log
        # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
        # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"get-service-url","hypothesisId":"C","location":"helm_utils.py:365","message":"Using fallback service name","data":{"service_name":service_name,"port":8000},"timestamp":int(time.time()*1000)})+"\n")
        # DEBUG DISABLED: #endregion
        return f"http://{service_name}.{namespace}.svc.cluster.local:8000/v1"  # Use service port 8000, not target port

    except ApiException as e:
        logger.error(
            f"Kubernetes API error while getting service URL for release '{release_name}' in namespace '{namespace}': "
            f"status={e.status}, reason={e.reason}, message={e.body if hasattr(e, 'body') else str(e)}"
        )
        raise RuntimeError(
            f"Failed to get service URL for release {release_name} in namespace {namespace}: "
            f"Kubernetes API returned status {e.status} ({e.reason})"
        )


def create_readiness_check_job(service_url: str, namespace: str, job_name: str, kubeconfig: Optional[str] = None) -> str:
    """Create a Kubernetes Job to check service readiness from inside the cluster.

    Args:
        service_url: Service URL to check (e.g., http://172.30.243.32:80/v1/models)
        namespace: Kubernetes namespace
        job_name: Name for the readiness check job
        kubeconfig: Path to kubeconfig file

    Returns:
        Job name
    """
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.error("kubernetes library not available")
        raise RuntimeError("kubernetes library required for Helm backend")

    try:
        if kubeconfig:
            config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
    except Exception as e:
        logger.error(f"Failed to load Kubernetes config: {e}")
        raise RuntimeError(f"Failed to load Kubernetes config: {e}")

    batch_v1 = client.BatchV1Api()

    # Create a readiness check job that retries with logging
    # Use /v1/models endpoint as recommended in llm-d docs
    check_url = service_url
    if not check_url.endswith("/v1/models"):
        # Ensure we're checking the models endpoint
        if "/v1" in check_url:
            check_url = check_url.replace("/v1", "/v1/models")
        else:
            check_url = f"{check_url}/v1/models"

    # Check for debug mode environment variable
    debug_mode = os.getenv("AUTO_TUNE_VLLM_DEBUG_READINESS_CHECK", "").lower() in (
        "1",
        "true",
        "yes",
    )

    # Create a script that retries with logging
    # In debug mode, run indefinitely; otherwise run for up to 5 minutes (300 seconds) checking every 5 seconds
    if debug_mode:
        # Debug mode: infinite retries, no timeout
        check_script = f"""#!/bin/sh
set -e
INTERVAL=5
ATTEMPT=0
URL="{check_url}"

echo "Starting readiness check for URL: $URL (DEBUG MODE - no timeout)"
echo "Will check every $INTERVAL seconds until service is ready"

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[Attempt $ATTEMPT] Checking $URL..."
    
    HTTP_CODE=$(curl -f -s -o /dev/null -w '%{{http_code}}' "$URL" || echo "000")
    echo "[Attempt $ATTEMPT] HTTP response code: $HTTP_CODE"
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Service is READY! HTTP 200 received."
        echo "READY"
        exit 0
    else
        echo "[Attempt $ATTEMPT] Service not ready yet (HTTP $HTTP_CODE), waiting $INTERVAL seconds..."
        sleep $INTERVAL
    fi
done
"""
        max_attempts = None  # No limit in debug mode
        active_deadline_seconds = None  # No timeout in debug mode
        ttl_seconds = 300  # Still clean up after 5 minutes
    else:
        # Normal mode: 60 attempts with 5 second intervals (5 minutes total)
        check_script = f"""#!/bin/sh
set -e
MAX_ATTEMPTS=60
INTERVAL=5
ATTEMPT=0
URL="{check_url}"

echo "Starting readiness check for URL: $URL"
echo "Will check up to $MAX_ATTEMPTS times with $INTERVAL second intervals"

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[Attempt $ATTEMPT/$MAX_ATTEMPTS] Checking $URL..."
    
    HTTP_CODE=$(curl -f -s -o /dev/null -w '%{{http_code}}' "$URL" || echo "000")
    echo "[Attempt $ATTEMPT/$MAX_ATTEMPTS] HTTP response code: $HTTP_CODE"
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Service is READY! HTTP 200 received."
        echo "READY"
        exit 0
    else
        echo "[Attempt $ATTEMPT/$MAX_ATTEMPTS] Service not ready yet (HTTP $HTTP_CODE), waiting $INTERVAL seconds..."
        sleep $INTERVAL
    fi
done

echo "Service did not become ready after $MAX_ATTEMPTS attempts"
echo "NOT_READY"
exit 1
"""
        max_attempts = 60
        active_deadline_seconds = 300  # Kill job after 5 minutes
        ttl_seconds = 300  # Clean up after 5 minutes

    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {
                "app": "auto-tune-vllm-readiness-check",
            },
        },
        "spec": {
            "backoffLimit": 0,  # No retries (we handle retries in the script)
            "ttlSecondsAfterFinished": ttl_seconds,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "auto-tune-vllm-readiness-check",
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "readiness-check",
                            "image": "quay.io/rh-ee-aansharm/curl:latest",  # Curl image from quay.io (avoids Docker Hub rate limits)
                            "command": ["/bin/sh", "-c"],
                            "args": [check_script],
                        },
                    ],
                },
            },
        },
    }

    # Only add activeDeadlineSeconds if not in debug mode
    if active_deadline_seconds is not None:
        job_manifest["spec"]["activeDeadlineSeconds"] = active_deadline_seconds

    if debug_mode:
        logger.info(
            f"DEBUG MODE ENABLED: Readiness check Job '{job_name}' will run "
            f"indefinitely (no timeout). Set AUTO_TUNE_VLLM_DEBUG_READINESS_CHECK=0 "
            f"to disable."
        )

    try:
        batch_v1.create_namespaced_job(namespace=namespace, body=job_manifest)
        logger.debug(f"Created readiness check Job: {job_name}")
        return job_name
    except ApiException as e:
        logger.error(
            f"Kubernetes API error while creating readiness check Job '{job_name}' in namespace '{namespace}': "
            f"status={e.status}, reason={e.reason}, message={e.body if hasattr(e, 'body') else str(e)}"
        )
        raise RuntimeError(
            f"Failed to create readiness check Job '{job_name}' in namespace '{namespace}': "
            f"Kubernetes API returned status {e.status} ({e.reason})"
        )


def check_readiness_job_result(job_name: str, namespace: str, kubeconfig: Optional[str] = None) -> bool:
    """Check if the readiness check Job succeeded.

    Args:
        job_name: Job name
        namespace: Kubernetes namespace
        kubeconfig: Path to kubeconfig file

    Returns:
        True if job succeeded, False otherwise
    """
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.error("kubernetes library not available")
        return False

    try:
        if kubeconfig:
            config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
    except Exception as e:
        logger.error(f"Failed to load Kubernetes config: {e}")
        return False

    batch_v1 = client.BatchV1Api()
    core_v1 = client.CoreV1Api()

    try:
        job = batch_v1.read_namespaced_job(name=job_name, namespace=namespace)

        if job.status.succeeded:
            # Job succeeded, check logs to confirm
            try:
                # Get pods for this job
                label_selector = f"job-name={job_name}"
                pods = core_v1.list_namespaced_pod(
                    namespace=namespace, label_selector=label_selector
                )

                if pods.items:
                    pod = pods.items[0]
                    pod_name = pod.metadata.name
                    logs = core_v1.read_namespaced_pod_log(
                        name=pod_name, namespace=namespace
                    )
                    # DEBUG DISABLED: #region agent log
                    # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                    # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"readiness-check","hypothesisId":"D","location":"helm_utils.py:685","message":"Readiness check Job logs","data":{"job_name":job_name,"logs":logs[:500]},"timestamp":int(time.time()*1000)})+"\n")
                    # DEBUG DISABLED: #endregion
                    if "READY" in logs:
                        logger.debug(
                            f"Readiness check Job {job_name} confirmed service is ready"
                        )
                        return True
                    elif "NOT_READY" in logs:
                        logger.debug(
                            f"Readiness check Job {job_name} confirmed service is not ready"
                        )
                        return False
            except Exception as e:
                logger.debug(
                    f"Error reading readiness check logs: {e}, but job succeeded so assuming ready"
                )
                return True  # Job succeeded, assume ready even if we can't read logs

            return True
        elif job.status.failed:
            logger.debug(f"Readiness check Job {job_name} failed")
            return False

        # Job still running
        return False
    except ApiException as e:
        if e.status == 404:
            logger.debug(f"Readiness check Job '{job_name}' not found")
            return False
        logger.error(
            f"Kubernetes API error while reading readiness check Job '{job_name}' in namespace '{namespace}': "
            f"status={e.status}, reason={e.reason}"
        )
        return False


def delete_readiness_check_job(job_name: str, namespace: str, kubeconfig: Optional[str] = None):
    """Delete the readiness check Job.

    Args:
        job_name: Job name
        namespace: Kubernetes namespace
        kubeconfig: Path to kubeconfig file
    """
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        return

    try:
        if kubeconfig:
            config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
    except Exception as e:
        logger.error(f"Failed to load Kubernetes config: {e}")
        return

    batch_v1 = client.BatchV1Api()

    try:
        batch_v1.delete_namespaced_job(
            name=job_name, namespace=namespace, propagation_policy="Background"
        )
        logger.debug(f"Deleted readiness check Job: {job_name}")
    except ApiException as e:
        if e.status != 404:  # Ignore if already deleted
            logger.debug(f"Error deleting readiness check Job '{job_name}': {e}")


def wait_for_service_ready(
    service_name: str,
    namespace: str,
    timeout: int = 300,
    kubeconfig: Optional[str] = None,
) -> bool:
    """Wait for Kubernetes service to be ready.

    Args:
        service_name: Service name or URL
        namespace: Kubernetes namespace
        timeout: Timeout in seconds

    Returns:
        True if service is ready, False if timeout
    """

    try:
        import requests
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.error("kubernetes or requests library not available")
        return False

    # DEBUG DISABLED: #region agent log
    # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
    # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"wait-service-ready","hypothesisId":"D","location":"helm_utils.py:360","message":"Starting wait_for_service_ready","data":{"service_name":service_name,"namespace":namespace,"timeout":timeout},"timestamp":int(time.time()*1000)})+"\n")
    # DEBUG DISABLED: #endregion

    try:
        if kubeconfig:
            config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
    except Exception as e:
        logger.error(f"Failed to load Kubernetes config: {e}")
        raise RuntimeError(f"Failed to load Kubernetes config: {e}")

    v1 = client.CoreV1Api()
    start_time = time.time()

    # Extract service name or IP from URL if full URL provided
    is_cluster_ip = False
    health_url = None
    service_obj = None

    if "://" in service_name:
        # Parse URL
        from urllib.parse import urlparse

        parsed = urlparse(service_name)
        hostname = parsed.hostname
        port = parsed.port or 8000

        # Check if it's a ClusterIP (IP address format like 172.30.x.x)
        if (
            hostname
            and all(c.isdigit() or c == "." for c in hostname)
            and hostname.count(".") == 3
        ):
            is_cluster_ip = True
            # For ClusterIP, we can't connect from outside cluster
            # Instead, check pod readiness via Kubernetes API
            logger.debug(
                f"Detected ClusterIP service ({hostname}), will check pod readiness instead of HTTP connection"
            )
            health_url = None  # Will use pod readiness check instead
            # DEBUG DISABLED: #region agent log
            # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
            # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"wait-service-ready","hypothesisId":"D","location":"helm_utils.py:603","message":"Detected ClusterIP, setting health_url=None","data":{"hostname":hostname,"is_cluster_ip":True},"timestamp":int(time.time()*1000)})+"\n")
            # DEBUG DISABLED: #endregion
        else:
            # DNS name - try to get service object
            service_name_only = hostname.split(".")[0] if hostname else service_name
            try:
                service_obj = v1.read_namespaced_service(
                    name=service_name_only, namespace=namespace
                )
                if service_obj.spec.type == "ClusterIP" and service_obj.spec.cluster_ip:
                    # ClusterIP service - check pod readiness instead
                    is_cluster_ip = True
                    logger.debug(
                        f"Service '{service_name_only}' is ClusterIP, will check pod readiness instead of HTTP connection"
                    )
                    health_url = None
                    # DEBUG DISABLED: #region agent log
                    # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                    # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"wait-service-ready","hypothesisId":"D","location":"helm_utils.py:613","message":"Service is ClusterIP, setting health_url=None","data":{"service_name":service_name_only,"is_cluster_ip":True},"timestamp":int(time.time()*1000)})+"\n")
                    # DEBUG DISABLED: #endregion
                else:
                    # LoadBalancer or NodePort - can connect directly
                    health_url = f"http://{hostname}:{port}/v1/models"
            except ApiException as e:
                logger.warning(
                    f"Kubernetes API error while reading service '{service_name_only}' in namespace '{namespace}': "
                    f"status={e.status}, reason={e.reason}. Assuming ClusterIP and creating readiness check Job."
                )
                # If we can't read the service, assume it's ClusterIP and create readiness check Job
                is_cluster_ip = True
                health_url = None
            except Exception as e:
                logger.warning(
                    f"Error reading service '{service_name_only}': {e}. Assuming ClusterIP and creating readiness check Job."
                )
                # If we can't read the service, assume it's ClusterIP and create readiness check Job
                is_cluster_ip = True
                health_url = None
    else:
        # Service name only - get service object
        try:
            service_obj = v1.read_namespaced_service(
                name=service_name, namespace=namespace
            )
            port = 8000
            if service_obj.spec.ports:
                port = service_obj.spec.ports[0].port
            if service_obj.spec.type == "ClusterIP" and service_obj.spec.cluster_ip:
                # ClusterIP service - check pod readiness instead
                is_cluster_ip = True
                logger.debug(
                    f"Service '{service_name}' is ClusterIP, will check pod readiness instead of HTTP connection"
                )
                health_url = None
                # DEBUG DISABLED: #region agent log
                # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"wait-service-ready","hypothesisId":"D","location":"helm_utils.py:641","message":"Service is ClusterIP, setting health_url=None","data":{"service_name":service_name,"is_cluster_ip":True},"timestamp":int(time.time()*1000)})+"\n")
                # DEBUG DISABLED: #endregion
            else:
                # LoadBalancer or NodePort - can connect directly
                health_url = f"http://{service_name}.{namespace}.svc.cluster.local:{port}/v1/models"
        except ApiException as e:
            logger.debug(
                f"Kubernetes API error while reading service '{service_name}' in namespace '{namespace}': "
                f"status={e.status}, reason={e.reason}. Using DNS name fallback."
            )
            # Fallback to DNS name
            health_url = (
                f"http://{service_name}.{namespace}.svc.cluster.local:8000/v1/models"
            )
        except Exception as e:
            logger.debug(
                f"Error reading service '{service_name}': {e}. Using DNS name fallback."
            )
            # Fallback to DNS name
            health_url = (
                f"http://{service_name}.{namespace}.svc.cluster.local:8000/v1/models"
            )

    # DEBUG DISABLED: #region agent log
    # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
    # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"wait-service-ready","hypothesisId":"D","location":"helm_utils.py:390","message":"Determined health URL","data":{"health_url":health_url,"service_name":service_name,"is_cluster_ip":is_cluster_ip},"timestamp":int(time.time()*1000)})+"\n")
    # DEBUG DISABLED: #endregion

    # Extract service name for verification
    service_name_for_check = None
    if "://" in service_name:
        from urllib.parse import urlparse

        parsed = urlparse(service_name)
        hostname = parsed.hostname
        # Extract service name if it's a DNS name, not an IP
        if hostname and not (
            all(c.isdigit() or c == "." for c in hostname) and hostname.count(".") == 3
        ):
            service_name_for_check = hostname.split(".")[0] if hostname else None
        elif is_cluster_ip and service_obj:
            # For ClusterIP, use the service object we already have
            service_name_for_check = service_obj.metadata.name
    else:
        service_name_for_check = service_name

    # If we have a service object but no service_name_for_check, use it
    if service_obj and not service_name_for_check:
        service_name_for_check = service_obj.metadata.name

    # For ClusterIP services, create a readiness check Job once and poll it
    readiness_check_job_name = None
    if is_cluster_ip and health_url is None:
        import uuid

        readiness_check_job_name = f"readiness-check-{uuid.uuid4().hex[:8]}"
        # Kubernetes names must be <= 63 characters
        if len(readiness_check_job_name) > 63:
            readiness_check_job_name = readiness_check_job_name[:63].rstrip("-")

        # DEBUG DISABLED: #region agent log
        # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
        # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"wait-service-ready","hypothesisId":"D","location":"helm_utils.py:686","message":"Creating readiness check Job for ClusterIP service","data":{"service_url":service_name,"job_name":readiness_check_job_name},"timestamp":int(time.time()*1000)})+"\n")
        # DEBUG DISABLED: #endregion

        try:
            create_readiness_check_job(
                service_name, namespace, readiness_check_job_name, kubeconfig
            )
            logger.info(
                f"Created readiness check Job '{readiness_check_job_name}' for ClusterIP service"
            )
        except Exception as e:
            logger.error(f"Failed to create readiness check Job: {e}")
            readiness_check_job_name = None
            # If we can't create the Job, we can't check readiness for ClusterIP
            logger.error(
                "Cannot check ClusterIP service readiness without Job. Service may not be accessible."
            )
            return False

    try:
        while time.time() - start_time < timeout:
            try:
                # For ClusterIP services, check readiness via the Kubernetes Job we created
                if readiness_check_job_name:
                    if check_readiness_job_result(readiness_check_job_name, namespace, kubeconfig):
                        # Service is ready!
                        logger.debug(
                            f"Service ready: readiness check Job {readiness_check_job_name} succeeded"
                        )
                        # DEBUG DISABLED: #region agent log
                        # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                        # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"wait-service-ready","hypothesisId":"D","location":"helm_utils.py:700","message":"Service ready via readiness check Job","data":{"job_name":readiness_check_job_name},"timestamp":int(time.time()*1000)})+"\n")
                        # DEBUG DISABLED: #endregion
                        delete_readiness_check_job(readiness_check_job_name, namespace, kubeconfig)
                        return True

                # For non-ClusterIP services or if pod check failed, try HTTP connection
                if health_url:
                    try:
                        response = requests.get(health_url, timeout=5)
                        # DEBUG DISABLED: #region agent log
                        # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                        # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"wait-service-ready","hypothesisId":"D","location":"helm_utils.py:740","message":"Readiness check response","data":{"status_code":response.status_code,"url":health_url},"timestamp":int(time.time()*1000)})+"\n")
                        # DEBUG DISABLED: #endregion
                        # /v1/models returns 200 when server is ready
                        if response.status_code == 200:
                            # Verify it's a valid models response (should have JSON with "data" or "object" field)
                            try:
                                data = response.json()
                                if "data" in data or "object" in data:
                                    logger.debug(
                                        "Service ready: /v1/models returned valid response"
                                    )
                                    return True
                            except (ValueError, KeyError):
                                # If not JSON or unexpected format, still consider 200 as ready
                                logger.debug(
                                    "Service ready: /v1/models returned 200 (non-JSON response)"
                                )
                                return True
                    except Exception:
                        # DEBUG DISABLED: #region agent log
                        # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                        # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"wait-service-ready","hypothesisId":"D","location":"helm_utils.py:754","message":"Readiness check failed","data":{"error":str(e),"url":health_url},"timestamp":int(time.time()*1000)})+"\n")
                        # DEBUG DISABLED: #endregion
                        pass
            except Exception:
                # DEBUG DISABLED: #region agent log
                # DEBUG DISABLED: with open("/home/thibrahi/workspace/auto-tune/llm-d-integration/.cursor/debug.log", "a") as f:
                # DEBUG DISABLED: f.write(json.dumps({"sessionId":"debug-session","runId":"wait-service-ready","hypothesisId":"D","location":"helm_utils.py:761","message":"Exception in wait loop","data":{"error":str(e)},"timestamp":int(time.time()*1000)})+"\n")
                # DEBUG DISABLED: #endregion
                pass

            time.sleep(2)
    except KeyboardInterrupt:
        logger.warning(
            "KeyboardInterrupt received during service readiness check. Cleaning up readiness check job..."
        )
        # Cleanup readiness check job if it still exists
        if readiness_check_job_name:
            try:
                delete_readiness_check_job(readiness_check_job_name, namespace, kubeconfig)
                logger.info(
                    f"Cleaned up readiness check Job '{readiness_check_job_name}' after interrupt"
                )
            except Exception as cleanup_e:
                logger.error(
                    f"Error cleaning up readiness check Job '{readiness_check_job_name}': {cleanup_e}"
                )
        raise
    finally:
        # Cleanup readiness check job if it still exists
        if readiness_check_job_name:
            try:
                delete_readiness_check_job(readiness_check_job_name, namespace, kubeconfig)
            except Exception as cleanup_e:
                logger.debug(
                    f"Error cleaning up readiness check Job '{readiness_check_job_name}': {cleanup_e}"
                )

    return False


def _build_benchmark_env(
    trial_config: TrialConfig, benchmark_type: str
) -> List[Dict[str, str]]:
    """Build environment variables for benchmark job based on benchmark type."""
    env = []
    if benchmark_type == "guidellm":
        env.append(
            {
                "name": "GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL",
                "value": trial_config.benchmark_config.logging_level,
            }
        )
    elif benchmark_type == "mlperf":
        # Enable Python debugging and verbose output
        env.extend(
            [
                {"name": "PYTHONUNBUFFERED", "value": "1"},
                {"name": "PYTHONFAULTHANDLER", "value": "1"},
                {"name": "MPLCONFIGDIR", "value": "/tmp/matplotlib"},
                # HuggingFace cache directories for tokenizer initialization
                # Must match the writable cache directories created in the benchmark container
                {"name": "HF_HOME", "value": "/tmp/.cache/huggingface"},
                {
                    "name": "TRANSFORMERS_CACHE",
                    "value": "/tmp/.cache/huggingface/transformers",
                },
                # AWS credentials for MLflow S3 artifact uploads
                {
                    "name": "AWS_ACCESS_KEY_ID",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": "aws-credentials",
                            "key": "AWS_ACCESS_KEY_ID",
                        }
                    },
                },
                {
                    "name": "AWS_SECRET_ACCESS_KEY",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": "aws-credentials",
                            "key": "AWS_SECRET_ACCESS_KEY",
                        }
                    },
                },
                {
                    "name": "AWS_DEFAULT_REGION",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": "aws-credentials",
                            "key": "AWS_DEFAULT_REGION",
                        }
                    },
                },
            ]
        )
    return env


def create_benchmark_job(
    trial_config: TrialConfig,
    server_url: str,
    namespace: str,
    benchmark_image: Optional[str] = None,
    kubeconfig: Optional[str] = None,
    benchmark_pvc: Optional[str] = None,
    model_pvc: Optional[str] = None,
) -> str:
    """Create Kubernetes Job for benchmark execution.

    Args:
        trial_config: Trial configuration
        server_url: vLLM server URL
        namespace: Kubernetes namespace
        benchmark_image: Optional container image for benchmark
        kubeconfig: Optional path to kubeconfig file
        benchmark_pvc: PersistentVolumeClaim name for benchmark results storage (REQUIRED for k8s backend)
        model_pvc: PersistentVolumeClaim name for model and dataset storage (optional, for MLPerf datasets)

    Returns:
        Job name

    Raises:
        ValueError: If benchmark_pvc is not provided
    """
    if not benchmark_pvc:
        raise ValueError(
            "benchmark_pvc is required for Kubernetes backend. "
            "Please specify benchmark_pvc in your k8s configuration."
        )
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.error("kubernetes library not available")
        raise RuntimeError("kubernetes library required for Helm backend")

    try:
        if kubeconfig:
            config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
    except Exception as e:
        raise RuntimeError(f"Failed to load Kubernetes config: {e}")

    # Verify PVC exists before creating the job
    core_v1 = client.CoreV1Api()
    try:
        pvc = core_v1.read_namespaced_persistent_volume_claim(
            name=benchmark_pvc, namespace=namespace
        )
        logger.debug(f"Verified PVC {benchmark_pvc} exists in namespace {namespace}")
    except ApiException as e:
        if e.status == 404:
            raise ValueError(
                f"PersistentVolumeClaim '{benchmark_pvc}' not found in namespace '{namespace}'. "
                f"Please create the PVC before running trials. "
                f"Example: kubectl create -f <pvc-manifest.yaml> -n {namespace}"
            )
        else:
            logger.warning(
                f"Could not verify PVC {benchmark_pvc} exists: {e}. "
                f"Proceeding with job creation, but pod may fail to start if PVC is missing."
            )
    except Exception as e:
        logger.warning(
            f"Could not verify PVC {benchmark_pvc} exists: {e}. "
            f"Proceeding with job creation, but pod may fail to start if PVC is missing."
        )

    batch_v1 = client.BatchV1Api()

    # Generate job name
    job_name = sanitize_release_name(
        f"{trial_config.study_name}-{trial_config.trial_id}-benchmark"
    )
    # Kubernetes names must be <= 63 characters
    if len(job_name) > 63:
        job_name = job_name[:63].rstrip("-")

    # Create appropriate benchmark provider based on benchmark type
    from ..benchmarks.providers import GuideLLMBenchmark, MLPerfBenchmark

    benchmark_type = trial_config.benchmark_config.benchmark_type
    if benchmark_type == "guidellm":
        benchmark_provider = GuideLLMBenchmark()
        default_image = "ghcr.io/vllm-project/guidellm:v0.5.1"
        working_dir: Optional[str] = None
    elif benchmark_type == "mlperf":
        benchmark_provider = MLPerfBenchmark()
        default_image = "quay.io/rh-ee-nmiriyal/mlperf-6.0:harness"
        working_dir = "/vllm-workspace/mlperf-inference-6.0-redhat"
    else:
        raise ValueError(
            f"Unsupported benchmark type for Helm execution: {benchmark_type}"
        )

    benchmark_provider.set_trial_context(trial_config.study_name, trial_config.trial_id)

    # Construct hierarchical results file path for shared PVC
    # Format: {study_name}/trial_{trial_id}/benchmark-results.json
    # PVC is required - no fallback to emptyDir
    results_base_path = "/mnt/results"
    study_name_safe = sanitize_release_name(trial_config.study_name)
    trial_id_safe = sanitize_release_name(trial_config.trial_id)
    results_dir = f"{results_base_path}/{study_name_safe}/trial_{trial_id_safe}"
    results_file = f"{results_dir}/benchmark-results.json"

    # Use trial-specific results_dir as output_dir to prevent log accumulation
    # This ensures MLPerf logs are isolated per trial (existing pattern from Thameem)
    if benchmark_type == "mlperf":
        logger.info(f"Using trial-specific output directory: {results_dir}")
        trial_config.benchmark_config.output_dir = results_dir

    # Build command args based on benchmark type
    if benchmark_type == "guidellm":
        cmd = benchmark_provider._build_guidellm_command(
            server_url, trial_config.benchmark_config, results_file
        )
    elif benchmark_type == "mlperf":
        cmd = benchmark_provider._build_mlperf_command(
            server_url, trial_config.benchmark_config, results_file
        )
    else:
        raise ValueError(f"Unsupported benchmark type: {benchmark_type}")

    # Use default image if benchmark_image is None or empty, ensuring we always use the correct image for the benchmark type
    # This prevents accidentally using the vLLM image for benchmark jobs
    if benchmark_image:
        # Safety check: warn if vLLM image is being used for benchmark
        if "vllm" in benchmark_image.lower() and "openai" in benchmark_image.lower():
            logger.warning(
                f"WARNING: Provided benchmark_image '{benchmark_image}' appears to be a vLLM image. "
                f"Using default benchmark image '{default_image}' instead for {benchmark_type} benchmark."
            )
            final_image = default_image
        else:
            final_image = benchmark_image
            logger.info(f"Using provided benchmark image: {final_image}")
    else:
        final_image = default_image
        logger.info(
            f"Using default benchmark image for {benchmark_type}: {final_image}"
        )

    # Create Job manifest
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {
                "app": "auto-tune-vllm-benchmark",
                "study-name": trial_config.study_name,
                "trial-id": trial_config.trial_id,
                "benchmark-type": benchmark_type,
            },
        },
        "spec": {
            "backoffLimit": 0,  # No retries
            "ttlSecondsAfterFinished": 300,  # Clean up after 5 minutes
            "template": {
                "metadata": {
                    "labels": {
                        "app": "auto-tune-vllm-benchmark",
                        "study-name": trial_config.study_name,
                        "trial-id": trial_config.trial_id,
                        "benchmark-type": benchmark_type,
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "initContainers": [
                        {
                            "name": "create-results-dir",
                            "image": final_image,  # Use benchmark image to avoid Docker Hub rate limits
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["sh", "-c", f"mkdir -p {results_dir}"],
                            "volumeMounts": [
                                {
                                    "name": "results",
                                    "mountPath": "/mnt/results",
                                },
                            ],
                        },
                    ],
                    "containers": [
                        {
                            "name": "benchmark",
                            "image": final_image,
                            "imagePullPolicy": "IfNotPresent",
                            **({"workingDir": working_dir} if working_dir else {}),
                            "command": ["/bin/bash", "-c"]
                            if benchmark_type == "mlperf"
                            else ([cmd[0]] if cmd else ["guidellm"]),
                            "args": [
                                f'''
echo "[DEBUG 1/12] Starting benchmark wrapper at $(date)"
echo "[DEBUG 2/12] Checking dataset file..."
ls -lh {trial_config.benchmark_config.dataset_path} 2>&1 || echo "[WARNING] Dataset file not found at {trial_config.benchmark_config.dataset_path}"
echo "[DEBUG 3/12] Testing vLLM server connectivity..."
python3 -c "import requests; r=requests.get('{server_url}/v1/models', timeout=5); print('vLLM OK' if r.status_code==200 else 'vLLM returned '+str(r.status_code))" 2>&1 || echo "[ERROR] vLLM unreachable"
echo "[DEBUG 4/12] Checking MLflow connectivity..."
python3 -c "import requests; r=requests.get('http://{trial_config.benchmark_config.mlflow_host}:5000/health', timeout=5); print('MLflow OK')" 2>&1 || echo "[WARNING] MLflow unreachable"
echo "[DEBUG 5/12] Python environment check..."
python3 --version
echo "[DEBUG 6/12] Checking working directory: $(pwd)"
ls -la harness/ | head -10
echo "[DEBUG 7/12] Verifying dataset is readable..."
python3 -c "
import os
dataset_path = '{trial_config.benchmark_config.dataset_path}'
if dataset_path.endswith('.json'):
    import json
    with open(dataset_path) as f:
        data = json.load(f)
    print(f'Dataset OK: {{len(data)}} samples')
elif dataset_path.endswith('.parquet'):
    import pandas as pd
    df = pd.read_parquet(dataset_path)
    print(f'Dataset OK: {{len(df)}} samples')
else:
    print(f'Dataset file exists: {{os.path.exists(dataset_path)}}')
" 2>&1 || echo "[ERROR] Dataset not readable"
echo "[DEBUG 8/12] Starting harness: {" ".join(cmd)}"
{" ".join(cmd)} 2>&1 &
PID=$!
echo "[DEBUG 9/12] Harness launched with PID: $PID"
echo "[DEBUG 10/12] Monitoring harness (status every 30s)..."
COUNT=0
while kill -0 $PID 2>/dev/null; do
    COUNT=$((COUNT+1))
    echo "[DEBUG] Harness running for $((COUNT*30))s ($(date))"
    sleep 30
done
wait $PID
EXIT_CODE=$?
echo "[DEBUG 11/12] Harness exited with code: $EXIT_CODE"

# Parse MLPerf results and create benchmark-results.json
if [ $EXIT_CODE -eq 0 ]; then
    echo "[DEBUG] Parsing MLPerf results..."
    python3 -c '
import json
import re
import sys
from pathlib import Path

# Parse MLPerf summary file
output_dir = Path("{trial_config.benchmark_config.output_dir}")
summary_file = output_dir / "mlperf" / "mlperf_log_summary.txt"

if not summary_file.exists():
    print(f"[ERROR] MLPerf summary file not found: {{summary_file}}")
    sys.exit(1)

# Read and print summary file for debugging
print("[DEBUG] Reading MLPerf summary file...")
with open(summary_file) as f:
    summary_content = f.read()

print("[DEBUG] ===== MLPerf Summary File Content =====")
print(summary_content)
print("[DEBUG] ===== End of Summary File =====")

# Extract metrics
metrics = dict()
errors_found = False

for line in summary_content.splitlines():
    line_lower = line.lower().strip()

    # Check for errors
    if "error encountered" in line_lower:
        errors_found = True
        print(f"[WARNING] MLPerf reported errors: {{line}}")

    # Extract: "Tokens per second: 307.598" (Offline) or "Completed tokens per second: 6094.77" (Server)
    # Handle both "tokens per second" and "completed tokens per second"
    if ("tokens per second" in line_lower) and ":" in line:
        parts = line.split(":", 1)
        if len(parts) == 2:
            try:
                value = float(parts[1].strip())
                metrics["output_tokens_per_second"] = value
                print(f"[DEBUG] Found tokens/sec: {{value}}")
            except ValueError as e:
                print(f"[DEBUG] Could not parse value from: {{line}} - {{e}}")

    # Extract: "Samples per second: 102.533" (Offline) or "Completed samples per second: 4.41" (Server)
    # Handle both "samples per second" and "completed samples per second"
    if ("samples per second" in line_lower) and ":" in line:
        parts = line.split(":", 1)
        if len(parts) == 2:
            try:
                value = float(parts[1].strip())
                metrics["samples_per_second"] = value
                print(f"[DEBUG] Found samples/sec: {{value}}")
            except ValueError as e:
                print(f"[DEBUG] Could not parse value from: {{line}} - {{e}}")

if errors_found:
    print("[ERROR] MLPerf benchmark encountered errors - results may be invalid")

if "output_tokens_per_second" not in metrics:
    print("[ERROR] Could not extract output_tokens_per_second from MLPerf summary")
    print(f"[ERROR] Metrics found: {{list(metrics.keys())}}")
    sys.exit(1)

# Write to results PVC
# Apply same sanitization as extract_job_results (sanitize_release_name)
def sanitize_name(name):
    # Convert to lowercase
    name = name.lower()
    # Replace underscores and spaces with hyphens
    name = name.replace("_", "-").replace(" ", "-")
    # Remove invalid characters (keep only alphanumeric and hyphens)
    name = re.sub(r"[^a-z0-9-]", "", name)
    # Remove consecutive hyphens
    name = re.sub(r"-+", "-", name)
    # Remove leading/trailing hyphens
    name = name.strip("-")
    # Ensure it starts with a letter or number
    if name and not name[0].isalnum():
        name = "release-" + name
    # Helm release names must be <= 53 characters
    if len(name) > 53:
        name = name[:53].rstrip("-")
    return name or "release"

study_name_safe = sanitize_name("{trial_config.study_name}")
trial_id_safe = sanitize_name("{trial_config.trial_id}")
results_dir = Path(f"/mnt/results/{{study_name_safe}}/trial_{{trial_id_safe}}")
results_dir.mkdir(parents=True, exist_ok=True)
results_file = results_dir / "benchmark-results.json"

with open(results_file, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"[INFO] Results written to {{results_file}}")
print(f"[INFO] Absolute path: {{results_file.absolute()}}")
print(f"[INFO] File exists: {{results_file.exists()}}")
print(f"[INFO] File size: {{results_file.stat().st_size}} bytes")
print("[INFO] output_tokens_per_second: " + str(metrics.get("output_tokens_per_second", 0)))
if "samples_per_second" in metrics:
    print("[INFO] samples_per_second: " + str(metrics.get("samples_per_second", 0)))

# Print JSON to logs for debugging and fallback parsing
print("[INFO] Benchmark results JSON:")
print(json.dumps(metrics, indent=2))
'
    PARSE_EXIT=$?
    if [ $PARSE_EXIT -ne 0 ]; then
        echo "[ERROR] Failed to parse MLPerf results"
        exit $PARSE_EXIT
    fi
fi

echo "[DEBUG 12/12] Complete at $(date)"
exit $EXIT_CODE
'''
                            ]
                            if benchmark_type == "mlperf"
                            else (cmd[1:] if len(cmd) > 1 else []),
                            "env": _build_benchmark_env(trial_config, benchmark_type),
                            "volumeMounts": [
                                {
                                    "name": "results",
                                    "mountPath": "/mnt/results",
                                },
                            ]
                            + (
                                [
                                    {
                                        "name": "guidellm-cache",
                                        "mountPath": "/home/guidellm/.cache",
                                    },
                                ]
                                if benchmark_type == "guidellm"
                                else []
                            )
                            + (
                                [
                                    {
                                        "name": "models",
                                        "mountPath": "/mnt/models",
                                    },
                                ]
                                if model_pvc
                                else []
                            ),
                        },
                        {
                            "name": "results-retriever",
                            "image": final_image,  # Use same image as benchmark (already pulled)
                            "command": ["sh", "-c", "sleep infinity"],
                            "volumeMounts": [
                                {
                                    "name": "results",
                                    "mountPath": "/mnt/results",
                                },
                            ]
                            + (
                                [
                                    {
                                        "name": "guidellm-cache",
                                        "mountPath": "/home/guidellm/.cache",
                                    },
                                ]
                                if benchmark_type == "guidellm"
                                else []
                            )
                            + (
                                [
                                    {
                                        "name": "models",
                                        "mountPath": "/mnt/models",
                                    },
                                ]
                                if model_pvc
                                else []
                            ),
                        },
                    ],
                    "volumes": [
                        {
                            "name": "results",
                            "persistentVolumeClaim": {
                                "claimName": benchmark_pvc,
                            },
                        },
                    ]
                    + (
                        [
                            {
                                "name": "guidellm-cache",
                                "emptyDir": {},
                            },
                        ]
                        if benchmark_type == "guidellm"
                        else []
                    )
                    + (
                        [
                            {
                                "name": "models",
                                "persistentVolumeClaim": {
                                    "claimName": model_pvc,
                                },
                            },
                        ]
                        if model_pvc
                        else []
                    ),
                },
            },
        },
    }

    try:
        # Create Job
        batch_v1.create_namespaced_job(namespace=namespace, body=job_manifest)
        logger.info(f"Created benchmark Job: {job_name}")
        return job_name
    except ApiException as e:
        logger.error(
            f"Kubernetes API error while creating benchmark Job '{job_name}' in namespace '{namespace}': "
            f"status={e.status}, reason={e.reason}, message={e.body if hasattr(e, 'body') else str(e)}"
        )
        raise RuntimeError(
            f"Failed to create benchmark Job '{job_name}' in namespace '{namespace}': "
            f"Kubernetes API returned status {e.status} ({e.reason})"
        )


def wait_for_job_completion(
    job_name: str, namespace: str, timeout: int = 3600, kubeconfig: Optional[str] = None
) -> bool:
    """Wait for Kubernetes Job to complete.

    Args:
        job_name: Job name
        namespace: Kubernetes namespace
        timeout: Timeout in seconds
        kubeconfig: Optional path to kubeconfig file

    Returns:
        True if job completed successfully, False if failed or timeout
    """
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.error("kubernetes library not available")
        return False

    try:
        if kubeconfig:
            config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
    except Exception as e:
        logger.error(f"Failed to load Kubernetes config: {e}")
        return False

    batch_v1 = client.BatchV1Api()
    core_v1 = client.CoreV1Api()
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            job = batch_v1.read_namespaced_job(name=job_name, namespace=namespace)

            if job.status.succeeded:
                logger.info(f"Job {job_name} completed successfully")
                return True
            elif job.status.failed:
                logger.error(f"Job {job_name} failed")
                # Collect logs for failed job
                try:
                    logs = collect_job_logs(job_name, namespace, kubeconfig=kubeconfig)
                    logger.error(f"Job {job_name} logs:\n{logs}")
                except Exception as log_e:
                    logger.warning(
                        f"Failed to collect logs for failed job {job_name}: {log_e}"
                    )
                return False

            # Check if pods are stuck in PodInitializing (likely PVC mounting issue)
            # Also check if benchmark container has completed (even if sidecar is still running)
            if job.status.active and job.status.active > 0:
                label_selector = f"job-name={job_name}"
                pods = core_v1.list_namespaced_pod(
                    namespace=namespace, label_selector=label_selector
                )
                for pod in pods.items:
                    if pod.status and pod.status.phase == "Pending":
                        # Check for PVC mounting issues
                        if pod.status.conditions:
                            for condition in pod.status.conditions:
                                if (
                                    condition.type == "PodScheduled"
                                    and condition.status != "True"
                                ):
                                    if condition.reason == "Unschedulable":
                                        logger.warning(
                                            f"Pod {pod.metadata.name} is unschedulable: {condition.message}. "
                                            f"This may indicate a PVC mounting issue. "
                                            f"Verify that PVC exists and is accessible in namespace {namespace}."
                                        )
                        # Check container status for init container issues
                        if pod.status.init_container_statuses:
                            for init_status in pod.status.init_container_statuses:
                                if init_status.state and init_status.state.waiting:
                                    if init_status.state.waiting.reason in [
                                        "ImagePullBackOff",
                                        "ErrImagePull",
                                        "CreateContainerError",
                                    ]:
                                        logger.warning(
                                            f"Init container {init_status.name} in pod {pod.metadata.name} "
                                            f"has issue: {init_status.state.waiting.reason} - {init_status.state.waiting.message}"
                                        )
                    elif pod.status and pod.status.phase == "Running":
                        # Check if benchmark container has completed successfully
                        # (Job won't show as succeeded if sidecar is still running)
                        if pod.status.container_statuses:
                            benchmark_completed = False
                            benchmark_failed = False
                            for container_status in pod.status.container_statuses:
                                if container_status.name == "benchmark":
                                    if (
                                        container_status.state
                                        and container_status.state.terminated
                                    ):
                                        if (
                                            container_status.state.terminated.exit_code
                                            == 0
                                        ):
                                            benchmark_completed = True
                                            logger.info(
                                                f"Benchmark container in pod {pod.metadata.name} completed successfully. "
                                                f"Sidecar may still be running for results retrieval."
                                            )
                                        else:
                                            benchmark_failed = True
                                            logger.error(
                                                f"Benchmark container in pod {pod.metadata.name} failed with exit code "
                                                f"{container_status.state.terminated.exit_code}"
                                            )
                                        break

                            if benchmark_completed:
                                # Benchmark is done, return True even though Job shows as active
                                logger.info(
                                    f"Job {job_name} benchmark completed successfully "
                                    f"(sidecar still running for results retrieval)"
                                )
                                return True
                            elif benchmark_failed:
                                # Benchmark failed, return False
                                logger.error(
                                    f"Job {job_name} benchmark container failed"
                                )
                                return False

            # Job still running
            time.sleep(5)
        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"Kubernetes API error while reading Job '{job_name}' in namespace '{namespace}': "
                    f"Job not found (status={e.status}, reason={e.reason})"
                )
                return False
            logger.error(
                f"Kubernetes API error while reading Job '{job_name}' in namespace '{namespace}': "
                f"status={e.status}, reason={e.reason}, message={e.body if hasattr(e, 'body') else str(e)}"
            )
            raise

    logger.warning(f"Job {job_name} timed out after {timeout}s")
    # Collect logs for timed out job
    try:
        logs = collect_job_logs(job_name, namespace, kubeconfig=kubeconfig)
        logger.warning(f"Job {job_name} logs (timeout):\n{logs}")
    except Exception as log_e:
        logger.warning(f"Failed to collect logs for timed out job {job_name}: {log_e}")
    return False


def collect_job_logs(
    job_name: str, namespace: str, kubeconfig: Optional[str] = None
) -> str:
    """Collect logs from all pods belonging to a Kubernetes Job.

    Args:
        job_name: Job name
        namespace: Kubernetes namespace
        kubeconfig: Optional path to kubeconfig file

    Returns:
        Combined logs from all pods as a string
    """
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.error("kubernetes library not available")
        return "Kubernetes library not available, cannot collect logs"

    try:
        if kubeconfig:
            config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
    except Exception as e:
        logger.error(f"Failed to load Kubernetes config: {e}")
        return f"Failed to load Kubernetes config: {e}"

    core_v1 = client.CoreV1Api()
    batch_v1 = client.BatchV1Api()

    all_logs = []

    try:
        # Get Job to check status
        try:
            job = batch_v1.read_namespaced_job(name=job_name, namespace=namespace)
            all_logs.append("=== Job Status ===\n")
            all_logs.append(f"Job: {job_name}\n")
            all_logs.append(f"Namespace: {namespace}\n")
            if job.status:
                all_logs.append(f"Succeeded: {job.status.succeeded}\n")
                all_logs.append(f"Failed: {job.status.failed}\n")
                all_logs.append(f"Active: {job.status.active}\n")
                if job.status.conditions:
                    for condition in job.status.conditions:
                        all_logs.append(
                            f"Condition: {condition.type} - {condition.status} - {condition.message or ''}\n"
                        )
            all_logs.append("\n")
        except ApiException as e:
            all_logs.append(f"Failed to read Job status: {e}\n\n")

        # Get pods for this job
        label_selector = f"job-name={job_name}"
        try:
            pods = core_v1.list_namespaced_pod(
                namespace=namespace, label_selector=label_selector
            )

            if not pods.items:
                all_logs.append(f"No pods found for Job {job_name}\n")
                return "".join(all_logs)

            # Collect logs from each pod
            for pod in pods.items:
                pod_name = pod.metadata.name
                all_logs.append(f"\n=== Pod: {pod_name} ===\n")

                # Add pod status
                if pod.status:
                    all_logs.append(f"Phase: {pod.status.phase}\n")
                    if pod.status.container_statuses:
                        for container_status in pod.status.container_statuses:
                            all_logs.append(f"Container: {container_status.name}\n")
                            if container_status.state:
                                if container_status.state.waiting:
                                    all_logs.append(
                                        f"  Waiting: {container_status.state.waiting.reason} - {container_status.state.waiting.message or ''}\n"
                                    )
                                if container_status.state.terminated:
                                    all_logs.append(
                                        f"  Terminated: {container_status.state.terminated.reason} - Exit Code: {container_status.state.terminated.exit_code}\n"
                                    )
                                    if container_status.state.terminated.message:
                                        all_logs.append(
                                            f"  Message: {container_status.state.terminated.message}\n"
                                        )
                    all_logs.append("\n")

                # Try to get logs from each container
                containers_to_check = ["benchmark"]
                if pod.spec and pod.spec.containers:
                    containers_to_check = [c.name for c in pod.spec.containers]

                for container_name in containers_to_check:
                    try:
                        logs = core_v1.read_namespaced_pod_log(
                            name=pod_name,
                            namespace=namespace,
                            container=container_name,
                            tail_lines=1000,  # Limit to last 1000 lines
                        )
                        all_logs.append(f"--- Container: {container_name} Logs ---\n")
                        all_logs.append(logs)
                        all_logs.append("\n")
                    except ApiException as e:
                        if e.status == 404:
                            all_logs.append(
                                f"Container {container_name} not found or pod not ready\n"
                            )
                        else:
                            all_logs.append(
                                f"Failed to read logs from container {container_name}: {e}\n"
                            )
                    except Exception as e:
                        all_logs.append(
                            f"Error reading logs from container {container_name}: {e}\n"
                        )

        except ApiException as e:
            all_logs.append(f"Failed to list pods for Job {job_name}: {e}\n")

    except Exception as e:
        all_logs.append(f"Error collecting logs: {e}\n")

    return "".join(all_logs)


def extract_job_results(
    job_name: str, namespace: str, kubeconfig: Optional[str] = None
) -> Dict[str, Any]:
    """Extract benchmark results from Kubernetes Job.

    Args:
        job_name: Job name
        namespace: Kubernetes namespace
        kubeconfig: Optional path to kubeconfig file

    Returns:
        Dictionary of benchmark results
    """
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.error("kubernetes library not available")
        raise RuntimeError("kubernetes library required for Helm backend")

    try:
        if kubeconfig:
            config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
    except Exception as e:
        raise RuntimeError(f"Failed to load Kubernetes config: {e}")

    core_v1 = client.CoreV1Api()
    batch_v1 = client.BatchV1Api()

    try:
        # Get Job to find pods and determine benchmark type
        job = batch_v1.read_namespaced_job(name=job_name, namespace=namespace)

        # Determine benchmark type from job labels
        benchmark_type = job.metadata.labels.get("benchmark-type", "guidellm")

        # Get pods for this job
        label_selector = f"job-name={job_name}"
        pods = core_v1.list_namespaced_pod(
            namespace=namespace, label_selector=label_selector
        )

        if not pods.items:
            raise RuntimeError(f"No pods found for Job {job_name}")

        # Get logs from the first pod
        pod = pods.items[0]
        pod_name = pod.metadata.name
        pod_phase = pod.status.phase if pod.status else "Unknown"

        # Extract study_name and trial_id from job labels for hierarchical path
        study_name = job.metadata.labels.get("study-name", "")
        trial_id = job.metadata.labels.get("trial-id", "")

        # Check if pod uses sidecar container (results-retriever)
        has_sidecar = False
        if pod.spec and pod.spec.containers:
            has_sidecar = any(
                c.name == "results-retriever" for c in pod.spec.containers
            )

        # Construct hierarchical results file path for PVC
        # Format: /mnt/results/{study_name}/trial_{trial_id}/benchmark-results.json
        study_name_safe = sanitize_release_name(study_name) if study_name else "unknown"
        trial_id_safe = sanitize_release_name(trial_id) if trial_id else "unknown"
        results_file_path = f"/mnt/results/{study_name_safe}/trial_{trial_id_safe}/benchmark-results.json"

        # Create appropriate benchmark provider
        from ..benchmarks.providers import GuideLLMBenchmark, MLPerfBenchmark

        if benchmark_type == "guidellm":
            benchmark_provider = GuideLLMBenchmark()
        elif benchmark_type == "mlperf":
            benchmark_provider = MLPerfBenchmark()
        else:
            logger.warning(
                f"Unknown benchmark type {benchmark_type}, defaulting to GuideLLM"
            )
            benchmark_provider = GuideLLMBenchmark()

        # Check if benchmark container is terminated
        benchmark_terminated = False
        if pod.status and pod.status.container_statuses:
            for container_status in pod.status.container_statuses:
                if container_status.name == "benchmark":
                    if container_status.state and container_status.state.terminated:
                        benchmark_terminated = True
                        break

        # Try to exec into pod and read the results file
        # If benchmark container is terminated and sidecar exists, use sidecar
        container_to_use = (
            "results-retriever"
            if (benchmark_terminated and has_sidecar)
            else "benchmark"
        )

        try:
            from kubernetes.stream import stream

            # First check if file exists
            check_command = ["test", "-f", results_file_path]
            try:
                check_resp = stream(
                    core_v1.connect_get_namespaced_pod_exec,
                    pod_name,
                    namespace,
                    command=check_command,
                    container=container_to_use,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                )
            except Exception:
                pass  # File might not exist, continue to try reading anyway

            # Read the file
            exec_command = ["cat", results_file_path]
            resp = stream(
                core_v1.connect_get_namespaced_pod_exec,
                pod_name,
                namespace,
                command=exec_command,
                container=container_to_use,
                stderr=False,  # Don't capture stderr to avoid mixing with stdout
                stdin=False,
                stdout=True,
                tty=False,
            )

            if resp:
                # Handle stream response - it might be a string or have channel info
                # Extract the actual content if it's a tuple or has channel prefixes
                content = resp
                logger.debug(
                    f"Stream resp type: {type(resp)}, length: {len(resp) if isinstance(resp, str) else 'N/A'}"
                )

                if isinstance(resp, tuple):
                    # Stream returns (stdout, stderr) tuple when both are enabled
                    content = resp[0] if resp[0] else ""
                    logger.debug(f"Extracted from tuple, content type: {type(content)}")

                # Strip whitespace
                content = content.strip()
                logger.debug(
                    f"Content after strip, length: {len(content)}, first 200 chars: {content[:200]}"
                )

                # Fix: Kubernetes stream API sometimes returns single quotes instead of double quotes
                # Convert single quotes to double quotes for valid JSON
                content = content.replace("'", '"')
                logger.debug(f"After quote fix, content: {content[:200]}")

                if content:
                    logger.debug(f"About to parse JSON, content: {content}")
                    try:
                        results_data = json.loads(content)
                        # Create temp file for parsing
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".json", delete=False
                        ) as f:
                            json.dump(results_data, f)
                            temp_file = f.name
                        try:
                            benchmark_provider._results_file = temp_file
                            results = benchmark_provider.parse_results()
                            return results
                        finally:
                            os.unlink(temp_file)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(
                            f"Failed to parse results from pod exec: {e}. "
                            f"Response preview (first 500 chars): {repr(content[:500])}"
                        )
        except Exception as e:
            logger.debug(
                f"Failed to exec into pod to read results file from container '{container_to_use}': {e}"
            )

        # Fallback: try to extract JSON from logs
        logs = core_v1.read_namespaced_pod_log(
            name=pod_name, namespace=namespace, container="benchmark"
        )

        # Try to find JSON results in logs (benchmarks may output JSON)
        lines = logs.split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    results_data = json.loads(line)
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".json", delete=False
                    ) as f:
                        json.dump(results_data, f)
                        temp_file = f.name
                    try:
                        benchmark_provider._results_file = temp_file
                        results = benchmark_provider.parse_results()
                        return results
                    finally:
                        os.unlink(temp_file)
                except (json.JSONDecodeError, KeyError):
                    continue

        # If we get here, we couldn't extract results
        raise RuntimeError(
            f"Could not extract benchmark results from Job {job_name}. "
            f"Results file not found in pod or logs. "
            f"Attempted to read from pod '{pod_name}' (phase: {pod_phase}) at path '{results_file_path}' "
            f"using container '{container_to_use}'. "
            f"Tried exec into {'sidecar' if (benchmark_terminated and has_sidecar) else 'benchmark'} container, then checked pod logs."
        )

    except ApiException as e:
        operation = (
            "reading Job"
            if "read_namespaced_job" in str(e)
            else "listing pods"
            if "list_namespaced_pod" in str(e)
            else "reading pod logs"
            if "read_namespaced_pod_log" in str(e)
            else "Kubernetes API operation"
        )
        logger.error(
            f"Kubernetes API error while {operation} for Job '{job_name}' in namespace '{namespace}': "
            f"status={e.status}, reason={e.reason}, message={e.body if hasattr(e, 'body') else str(e)}"
        )
        raise RuntimeError(
            f"Failed to extract results from Job '{job_name}' in namespace '{namespace}': "
            f"Kubernetes API returned status {e.status} ({e.reason}) during {operation}"
        )


def delete_benchmark_job(job_name: str, namespace: str) -> None:
    """Delete Kubernetes Job.

    Args:
        job_name: Job name
        namespace: Kubernetes namespace
    """
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.error("kubernetes library not available")
        return

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    batch_v1 = client.BatchV1Api()

    try:
        batch_v1.delete_namespaced_job(
            name=job_name,
            namespace=namespace,
            propagation_policy="Background",
        )
        logger.info(f"Deleted benchmark Job: {job_name}")
    except ApiException as e:
        if e.status == 404:
            logger.debug(
                f"Kubernetes API: Job '{job_name}' in namespace '{namespace}' not found (already deleted)"
            )
        else:
            logger.warning(
                f"Kubernetes API error while deleting Job '{job_name}' in namespace '{namespace}': "
                f"status={e.status}, reason={e.reason}, message={e.body if hasattr(e, 'body') else str(e)}"
            )
