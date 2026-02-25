"""Kubernetes utilities for direct Deployment/Pod-based vLLM deployment."""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, Optional

from ..core.trial import TrialConfig

logger = logging.getLogger(__name__)


def sanitize_k8s_name(name: str) -> str:
    """Sanitize name for Kubernetes resource (lowercase, alphanumeric and hyphens only).

    Args:
        name: Original name

    Returns:
        Sanitized name suitable for Kubernetes resource
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
        name = "resource-" + name
    # Kubernetes names must be <= 63 characters
    if len(name) > 63:
        name = name[:63].rstrip("-")
    return name or "resource"


def create_vllm_deployment(
    trial_config: TrialConfig,
    deployment_name: str,
    namespace: str,
    vllm_image: str,
    resource_requests: Optional[Dict[str, str]] = None,
    resource_limits: Optional[Dict[str, str]] = None,
    kubeconfig: Optional[str] = None,
    model_pvc: Optional[str] = None,
) -> str:
    """Create Kubernetes Deployment for vLLM server.

    Args:
        trial_config: Trial configuration
        deployment_name: Name for the Deployment
        namespace: Kubernetes namespace
        vllm_image: Container image for vLLM
        resource_requests: Resource requests (e.g., {"nvidia.com/gpu": "1"})
        resource_limits: Resource limits (e.g., {"nvidia.com/gpu": "1", "memory": "32Gi"})
        kubeconfig: Path to kubeconfig file
        model_pvc: PersistentVolumeClaim name for model storage (e.g., "model-pvc")

    Returns:
        Deployment name
    """
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.error("kubernetes library not available")
        raise RuntimeError("kubernetes library required for Kubernetes backend")

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

    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()

    # Build vLLM command and args
    vllm_args = trial_config.vllm_args
    env_vars = (
        trial_config.environment_vars.copy() if trial_config.environment_vars else {}
    )

    # Set HF_HOME to PVC mount path if model_pvc is specified
    model_mount_path = "/mnt/models"
    if model_pvc:
        env_vars["HF_HOME"] = model_mount_path
        # Set cache directories to writable PVC location to avoid OpenShift permission issues
        # flashinfer and vllm need writable cache directories
        env_vars["HOME"] = f"{model_mount_path}/.cache/home"
        env_vars["XDG_CACHE_HOME"] = f"{model_mount_path}/.cache"
        env_vars["FLASHINFER_WORKSPACE_DIR"] = f"{model_mount_path}/.cache/flashinfer"
        env_vars["VLLM_CACHE_ROOT"] = f"{model_mount_path}/.cache/vllm"

    # Build container command
    cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    args = vllm_args.copy()

    # Add model if not already in args
    model_in_args = any("--model" in arg for arg in args)
    if not model_in_args and trial_config.benchmark_config:
        args = ["--model", trial_config.benchmark_config.model] + args

    logger.info(f"vLLM arguments: {' '.join(args)}")

    try:
        # Create Deployment using proper Kubernetes client objects
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=deployment_name,
                namespace=namespace,
                labels={
                    "app": "vllm-server",
                    "study-name": sanitize_k8s_name(trial_config.study_name),
                    "trial-id": sanitize_k8s_name(trial_config.trial_id),
                },
            ),
            spec=client.V1DeploymentSpec(
                replicas=1,
                selector=client.V1LabelSelector(
                    match_labels={
                        "app": "vllm-server",
                        "trial-id": sanitize_k8s_name(trial_config.trial_id),
                    }
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={
                            "app": "vllm-server",
                            "study-name": sanitize_k8s_name(trial_config.study_name),
                            "trial-id": sanitize_k8s_name(trial_config.trial_id),
                        },
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="vllm-server",
                                image=vllm_image,
                                command=cmd,
                                args=args,
                                env=[
                                    client.V1EnvVar(name=k, value=str(v))
                                    for k, v in env_vars.items()
                                ]
                                if env_vars
                                else [],
                                ports=[
                                    client.V1ContainerPort(
                                        container_port=8000,
                                        name="http",
                                        protocol="TCP",
                                    )
                                ],
                                volume_mounts=[
                                    client.V1VolumeMount(
                                        name="cache",
                                        mount_path="/.cache",
                                    ),
                                    client.V1VolumeMount(
                                        name="config",
                                        mount_path="/.config",
                                    ),
                                    client.V1VolumeMount(
                                        name="triton",
                                        mount_path="/.triton",
                                    ),
                                ]
                                + (
                                    [
                                        client.V1VolumeMount(
                                            name="model-pvc",
                                            mount_path=model_mount_path,
                                        )
                                    ]
                                    if model_pvc
                                    else []
                                ),
                                resources=client.V1ResourceRequirements(
                                    requests=resource_requests or {},
                                    limits=resource_limits or {},
                                )
                                if (resource_requests or resource_limits)
                                else None,
                            )
                        ],
                        volumes=[
                            client.V1Volume(
                                name="cache",
                                empty_dir=client.V1EmptyDirVolumeSource(),
                            ),
                            client.V1Volume(
                                name="config",
                                empty_dir=client.V1EmptyDirVolumeSource(),
                            ),
                            client.V1Volume(
                                name="triton",
                                empty_dir=client.V1EmptyDirVolumeSource(),
                            ),
                        ]
                        + (
                            [
                                client.V1Volume(
                                    name="model-pvc",
                                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                        claim_name=model_pvc,
                                    ),
                                )
                            ]
                            if model_pvc
                            else []
                        ),
                        restart_policy="Always",
                    ),
                ),
            ),
        )
        apps_v1.create_namespaced_deployment(namespace=namespace, body=deployment)
        logger.info(
            f"Created vLLM Deployment: {deployment_name} in namespace {namespace}"
        )
        return deployment_name
    except ApiException as e:
        logger.error(
            f"Kubernetes API error creating Deployment '{deployment_name}': "
            f"status={e.status}, reason={e.reason}, message={e.body if hasattr(e, 'body') else str(e)}"
        )
        raise RuntimeError(
            f"Failed to create Deployment '{deployment_name}': "
            f"Kubernetes API returned status {e.status} ({e.reason})"
        )


def create_vllm_service(
    trial_config: TrialConfig,
    service_name: str,
    deployment_name: str,
    namespace: str,
    service_type: str = "ClusterIP",
    service_port: int = 8000,
    kubeconfig: Optional[str] = None,
) -> str:
    """Create Kubernetes Service for vLLM server.

    Args:
        trial_config: Trial configuration
        service_name: Name for the Service
        deployment_name: Name of the Deployment to target
        namespace: Kubernetes namespace
        service_type: Service type (ClusterIP, NodePort, LoadBalancer)
        service_port: Service port
        kubeconfig: Path to kubeconfig file

    Returns:
        Service name
    """
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.error("kubernetes library not available")
        raise RuntimeError("kubernetes library required for Kubernetes backend")

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

    try:
        # Create Service using proper Kubernetes client objects
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=service_name,
                namespace=namespace,
                labels={
                    "app": "vllm-server",
                    "study-name": sanitize_k8s_name(trial_config.study_name),
                    "trial-id": sanitize_k8s_name(trial_config.trial_id),
                },
            ),
            spec=client.V1ServiceSpec(
                type=service_type,
                selector={
                    "app": "vllm-server",
                    "trial-id": sanitize_k8s_name(trial_config.trial_id),
                },
                ports=[
                    client.V1ServicePort(
                        port=service_port,
                        target_port=8000,
                        protocol="TCP",
                        name="http",
                    )
                ],
            ),
        )
        core_v1.create_namespaced_service(namespace=namespace, body=service)
        logger.info(f"Created vLLM Service: {service_name} in namespace {namespace}")
        return service_name
    except ApiException as e:
        logger.error(
            f"Kubernetes API error creating Service '{service_name}': "
            f"status={e.status}, reason={e.reason}, message={e.body if hasattr(e, 'body') else str(e)}"
        )
        raise RuntimeError(
            f"Failed to create Service '{service_name}': "
            f"Kubernetes API returned status {e.status} ({e.reason})"
        )


def get_service_url(
    service_name: str,
    namespace: str,
    service_type: str = "ClusterIP",
    kubeconfig: Optional[str] = None,
) -> str:
    """Get URL for Kubernetes Service.

    Args:
        service_name: Service name
        namespace: Kubernetes namespace
        service_type: Service type
        kubeconfig: Path to kubeconfig file

    Returns:
        Service URL
    """
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.error("kubernetes library not available")
        raise RuntimeError("kubernetes library required for Kubernetes backend")

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

    try:
        service = core_v1.read_namespaced_service(
            name=service_name, namespace=namespace
        )

        if service_type == "LoadBalancer":
            # Wait for LoadBalancer IP
            max_wait = 300  # 5 minutes
            start_time = time.time()
            while time.time() - start_time < max_wait:
                service = core_v1.read_namespaced_service(
                    name=service_name, namespace=namespace
                )
                if (
                    service.status.load_balancer
                    and service.status.load_balancer.ingress
                ):
                    ingress = service.status.load_balancer.ingress[0]
                    ip = ingress.ip or ingress.hostname
                    if ip:
                        port = service.spec.ports[0].port
                        return f"http://{ip}:{port}"
                time.sleep(5)
            raise RuntimeError(f"LoadBalancer IP not assigned after {max_wait}s")

        elif service_type == "NodePort":
            # Get node IP (simplified - uses first node)
            nodes = core_v1.list_node()
            if not nodes.items:
                raise RuntimeError("No nodes found in cluster")
            node_ip = nodes.items[0].status.addresses[0].address
            node_port = service.spec.ports[0].node_port
            return f"http://{node_ip}:{node_port}"

        else:  # ClusterIP
            # Return cluster-internal URL
            port = service.spec.ports[0].port
            return f"http://{service_name}.{namespace}.svc.cluster.local:{port}"

    except ApiException as e:
        raise RuntimeError(f"Failed to get service URL: {e}")


def wait_for_deployment_ready(
    deployment_name: str,
    namespace: str,
    timeout: int = 600,
    kubeconfig: Optional[str] = None,
) -> bool:
    """Wait for Kubernetes Deployment to be ready.

    Args:
        deployment_name: Deployment name
        namespace: Kubernetes namespace
        timeout: Timeout in seconds
        kubeconfig: Path to kubeconfig file

    Returns:
        True if ready, False if timeout
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

    apps_v1 = client.AppsV1Api()
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name, namespace=namespace
            )

            if (
                deployment.status.ready_replicas is not None
                and deployment.status.ready_replicas >= 1
                and deployment.status.replicas == deployment.status.ready_replicas
            ):
                logger.info(f"Deployment {deployment_name} is ready")
                return True

            time.sleep(5)
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"Deployment {deployment_name} not found")
                return False
            logger.warning(f"Error checking deployment status: {e}")
            time.sleep(5)

    logger.warning(f"Deployment {deployment_name} not ready after {timeout}s")
    return False


def delete_vllm_resources(
    deployment_name: str,
    service_name: str,
    namespace: str,
    kubeconfig: Optional[str] = None,
) -> None:
    """Delete vLLM Deployment and Service.

    Args:
        deployment_name: Deployment name
        service_name: Service name
        namespace: Kubernetes namespace
        kubeconfig: Path to kubeconfig file
    """
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.error("kubernetes library not available")
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

    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()

    # Delete Service
    try:
        core_v1.delete_namespaced_service(name=service_name, namespace=namespace)
        logger.info(f"Deleted Service: {service_name}")
    except ApiException as e:
        if e.status != 404:
            logger.warning(f"Failed to delete Service {service_name}: {e}")

    # Delete Deployment
    try:
        apps_v1.delete_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            propagation_policy="Foreground",
        )
        logger.info(f"Deleted Deployment: {deployment_name}")
    except ApiException as e:
        if e.status != 404:
            logger.warning(f"Failed to delete Deployment {deployment_name}: {e}")
