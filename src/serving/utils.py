import os
import socket

try:
    from huggingface_hub import HfApi

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print(
        "Warning: huggingface_hub not available. Install with: pip install huggingface_hub"
    )


def validate_huggingface_model(model_name):
    if not HF_AVAILABLE:
        print(
            f"Warning: Cannot validate HuggingFace model '{model_name}' - huggingface_hub not installed"
        )
        return True

    try:
        api = HfApi()
        api.model_info(model_name)
        print(f"✓ Validated HuggingFace model: {model_name}")
        return True
    except Exception as e:
        print(f"✗ Invalid HuggingFace model '{model_name}': {str(e)}")
        return False


def log_stream(stream, log_file, prefix):
    with open(log_file, "a") as f:
        while True:
            line = stream.readline()
            if not line:
                break
            print(f"[{prefix}] {line.strip()}")
            f.write(line)
            f.flush()


def check_port_available(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", port))
        return True
    except socket.error:
        return False
    finally:
        sock.close()


def get_port_for_gpu(gpu_id, start_port):
    """Get port number for a specific GPU ID from prescribed range."""
    return start_port + gpu_id


def check_all_ports_available_for_study(gpu_ids, start_port):
    """
    Check if ALL required ports are available for parallel study.
    This must be called before study starts.

    Args:
        gpu_ids: List of GPU IDs that will be used
        start_port: Starting port number from user

    Returns:
        Tuple of (all_available: bool, unavailable_ports: List[int])
    """
    unavailable_ports = []
    required_ports = []

    for gpu_id in gpu_ids:
        port = get_port_for_gpu(gpu_id, start_port)
        required_ports.append(port)
        if not check_port_available(port):
            unavailable_ports.append(port)

    return len(unavailable_ports) == 0, unavailable_ports, required_ports


def get_last_log_lines(log_file, n=20):
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            return "".join(lines[-n:]) if lines else ""
    except Exception as e:
        return f"Could not read log file: {str(e)}"


def save_config_to_study(config_path, study_dir, study_id):
    """Save the configuration file to the study directory."""
    import shutil

    config_filename = f"vllm_config_study_{study_id}.yaml"
    dest_path = os.path.join(study_dir, config_filename)

    try:
        shutil.copy2(config_path, dest_path)
        print(f"✓ Configuration saved to: {dest_path}")
        return dest_path
    except Exception as e:
        print(f"✗ Failed to save configuration: {e}")
        return None
