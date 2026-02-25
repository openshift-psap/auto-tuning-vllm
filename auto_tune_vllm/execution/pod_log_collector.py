"""Pod log collection utilities for Kubernetes.

This module provides functions to collect logs from Kubernetes pods
(both Job pods and Deployment pods) and store them in PostgreSQL
before pod deletion.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def _get_kubernetes_client(kubeconfig: Optional[str] = None):
    """Get Kubernetes client API objects.

    Args:
        kubeconfig: Path to kubeconfig file (optional)

    Returns:
        Tuple of (CoreV1Api, BatchV1Api, AppsV1Api)

    Raises:
        RuntimeError: If Kubernetes client setup fails
    """
    try:
        from kubernetes import client, config
    except ImportError:
        raise RuntimeError("kubernetes library not available")

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

    return client.CoreV1Api(), client.BatchV1Api(), client.AppsV1Api()


def _store_pod_logs_to_postgres(
    logs: str,
    pod_name: str,
    container_name: str,
    study_name: str,
    trial_id: str,
    component: str,
    pg_url: str,
    worker_node: str = "k8s-pod",
) -> bool:
    """Store pod logs to PostgreSQL line-by-line in single transaction.

    Args:
        logs: Pod log content
        pod_name: Kubernetes pod name
        container_name: Container name within pod
        study_name: Study name for trial_logs table
        trial_id: Trial ID for trial_logs table
        component: Component name (e.g., 'benchmark-pod', 'vllm-pod')
        pg_url: PostgreSQL connection URL
        worker_node: Worker node identifier (default: 'k8s-pod')

    Returns:
        True if logs stored successfully, False otherwise
    """
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 not available, cannot store pod logs")
        return False

    if not logs or not logs.strip():
        logger.info(f"No logs to store for pod {pod_name}, container {container_name}")
        return True

    try:
        with psycopg2.connect(pg_url) as conn:
            with conn.cursor() as cur:
                timestamp = datetime.now()

                # Insert each line in single transaction
                for line in logs.split('\n'):
                    if line.strip():
                        cur.execute(
                            """
                            INSERT INTO trial_logs
                            (study_name, trial_id, component, timestamp, level, message,
                             worker_node, pod_name, container_name)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (study_name, trial_id, component, timestamp, 'INFO',
                             line.strip(), worker_node, pod_name, container_name)
                        )

                conn.commit()
                logger.info(
                    f"Stored logs for pod {pod_name}, container {container_name} "
                    f"({len(logs.split(chr(10)))} lines)"
                )
                return True

    except Exception as e:
        logger.warning(f"Failed to store pod logs to PostgreSQL: {e}")
        return False


def collect_and_store_job_pod_logs(
    job_name: str,
    namespace: str,
    study_name: str,
    trial_id: str,
    pg_url: str,
    kubeconfig: Optional[str] = None,
    component: str = "benchmark-pod",
) -> bool:
    """Collect logs from all pods of a Kubernetes Job and store to PostgreSQL.

    This function finds all pods associated with a Job using the label selector
    'job-name={job_name}' and collects logs from all containers in each pod.

    Args:
        job_name: Kubernetes Job name
        namespace: Kubernetes namespace
        study_name: Study name for trial_logs table
        trial_id: Trial ID for trial_logs table
        pg_url: PostgreSQL connection URL
        kubeconfig: Path to kubeconfig file (optional)
        component: Component name for logs (default: 'benchmark-pod')

    Returns:
        True if logs collected successfully from at least one pod, False otherwise
    """
    try:
        core_v1, _, _ = _get_kubernetes_client(kubeconfig)
    except RuntimeError as e:
        logger.warning(f"Failed to get Kubernetes client: {e}")
        return False

    try:
        # Find pods using label selector
        pods = core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"job-name={job_name}"
        )

        if not pods.items:
            logger.warning(f"No pods found for Job {job_name}")
            return False

        success_count = 0
        for pod in pods.items:
            pod_name = pod.metadata.name

            # Collect logs from all containers
            for container in pod.spec.containers:
                container_name = container.name

                try:
                    # Read logs from container
                    logs = core_v1.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=namespace,
                        container=container_name
                    )

                    # Store logs to PostgreSQL
                    if _store_pod_logs_to_postgres(
                        logs=logs,
                        pod_name=pod_name,
                        container_name=container_name,
                        study_name=study_name,
                        trial_id=trial_id,
                        component=component,
                        pg_url=pg_url,
                    ):
                        success_count += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to collect logs from pod {pod_name}, "
                        f"container {container_name}: {e}"
                    )

        return success_count > 0

    except Exception as e:
        logger.warning(f"Failed to collect Job pod logs: {e}")
        return False


def collect_and_store_deployment_pod_logs(
    deployment_name: str,
    trial_id: str,
    namespace: str,
    study_name: str,
    pg_url: str,
    kubeconfig: Optional[str] = None,
    component: str = "vllm-pod",
    container_name: str = "vllm-server",
) -> bool:
    """Collect logs from pods of a Kubernetes Deployment and store to PostgreSQL.

    This function finds all pods associated with a Deployment using the label
    selector 'app=vllm-server,trial-id={sanitized_trial_id}' and collects logs
    from the specified container.

    Args:
        deployment_name: Kubernetes Deployment name
        trial_id: Trial ID (will be sanitized for label selector)
        namespace: Kubernetes namespace
        study_name: Study name for trial_logs table
        pg_url: PostgreSQL connection URL
        kubeconfig: Path to kubeconfig file (optional)
        component: Component name for logs (default: 'vllm-pod')
        container_name: Container name to collect logs from (default: 'vllm-server')

    Returns:
        True if logs collected successfully from at least one pod, False otherwise
    """
    try:
        core_v1, _, _ = _get_kubernetes_client(kubeconfig)
    except RuntimeError as e:
        logger.warning(f"Failed to get Kubernetes client: {e}")
        return False

    try:
        # Import sanitization function
        from .k8s_utils import sanitize_k8s_name

        # Sanitize trial_id for label selector
        sanitized_trial_id = sanitize_k8s_name(trial_id)

        # Find pods using label selector
        pods = core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"app=vllm-server,trial-id={sanitized_trial_id}"
        )

        if not pods.items:
            logger.warning(
                f"No pods found for Deployment {deployment_name} "
                f"with trial-id={sanitized_trial_id}"
            )
            return False

        success_count = 0
        for pod in pods.items:
            pod_name = pod.metadata.name

            try:
                # Read logs from specified container
                logs = core_v1.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=namespace,
                    container=container_name
                )

                # Store logs to PostgreSQL
                if _store_pod_logs_to_postgres(
                    logs=logs,
                    pod_name=pod_name,
                    container_name=container_name,
                    study_name=study_name,
                    trial_id=trial_id,
                    component=component,
                    pg_url=pg_url,
                ):
                    success_count += 1

            except Exception as e:
                logger.warning(
                    f"Failed to collect logs from pod {pod_name}, "
                    f"container {container_name}: {e}"
                )

        return success_count > 0

    except Exception as e:
        logger.warning(f"Failed to collect Deployment pod logs: {e}")
        return False
