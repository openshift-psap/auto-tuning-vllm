import subprocess
import time
import requests
import os
import signal
import json
from src.serving.utils import check_port_available, get_last_log_lines

def build_vllm_command(model_name, port, candidate_flags):
    cmd = [
        "vllm",
        "serve",
        model_name,
        "--max-model-len", "4096",
        "--port", str(port),
        "--disable-log-requests"
    ]
    
    # Check if we need to add --enable-chunked-prefill based on --max-num-partial-prefills
    max_partial_prefills_value = None
    has_enable_chunked_prefill = False
    
    i = 0
    while i < len(candidate_flags):
        flag = candidate_flags[i]
        
        if flag == "--enable-chunked-prefill" or flag.startswith("--enable-chunked-prefill="):
            has_enable_chunked_prefill = True
        
        elif flag == "--max-num-partial-prefills":
            if i + 1 < len(candidate_flags):
                try:
                    max_partial_prefills_value = int(candidate_flags[i + 1])
                except ValueError:
                    pass
        elif flag.startswith("--max-num-partial-prefills="):
            try:
                value_str = flag.split("=", 1)[1]
                max_partial_prefills_value = int(value_str)
            except (ValueError, IndexError):
                pass
        
        i += 1
    
    if max_partial_prefills_value is not None and max_partial_prefills_value > 1 and not has_enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")
    
    cmd += candidate_flags
    return cmd

def start_vllm_server(cmd, ready_pattern="Application startup complete", timeout=30000, log_file=None, env=None):
    if not check_port_available(int(cmd[cmd.index('--port') + 1])):
        raise RuntimeError(f"Port {cmd[cmd.index('--port') + 1]} is already in use")

    print(f"Launching vLLM server {' '.join(cmd)}")
    
    # Set environment variables to help prevent CUDA out of memory issues
    if env is None:
        env = os.environ.copy()
    # env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # print("Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to prevent CUDA memory fragmentation")
    
    try:
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=f,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,
                env=env,
            )
    except FileNotFoundError:
        raise RuntimeError("vLLM binary not found. Is vllm installed and in PATH?")
    except OSError as e:
        raise RuntimeError(f"Failed to start vLLM server: {e.strerror}\nCommand: {' '.join(cmd)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error starting vLLM server: {str(e)}")

    start_time = time.time()
    server_ready = False

    while True:
        if proc.poll() is not None:
            last_logs = get_last_log_lines(log_file)
            raise RuntimeError(
                f"vLLM server exited unexpectedly with code {proc.returncode}.\n"
                f"Last log lines:\n{last_logs}\n"
                f"Check the full log file for details: {log_file}"
            )
        
        elapsed = time.time() - start_time
        if elapsed > timeout:
            proc.kill()
            raise TimeoutError(f"vLLM server did not start within {timeout/1000} seconds")

        if not server_ready:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if ready_pattern in content:
                        time.sleep(10)
                        server_ready = True
                        return proc
            except IOError as e:
                proc.kill()
                raise RuntimeError(f"Failed to read vLLM log file: {str(e)}")

        time.sleep(0.1)

def stop_vllm_server(proc, grace_period=15):
    if proc is None:
        print("No vLLM process to stop")
        return

    print(f"Stopping vLLM process {proc.pid}...")
    killed = False

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        print("Sent SIGINT to process group")
    except ProcessLookupError:
        print("Process already terminated")
        return
    except Exception as e:
        print(f"Warning: SIGINT failed: {str(e)}")
    
    print(f"Waiting up to {grace_period}s for graceful shutdown...")
    deadline = time.time() + grace_period
    while time.time() < deadline:
        if proc.poll() is not None:
            print("vLLM process exited cleanly")
            return
        time.sleep(0.5)

    if proc.poll() is None:
        print(f"Process {proc.pid} still alive after {grace_period}s; sending SIGTERM")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            print("Sent SIGTERM to process group")
            
            # Wait another 5 seconds for SIGTERM
            deadline = time.time() + 5
            while time.time() < deadline:
                if proc.poll() is not None:
                    print("vLLM process exited after SIGTERM")
                    return
                time.sleep(0.5)
                
        except ProcessLookupError:
            print("Process already terminated")
            return
        except Exception as e:
            print(f"Warning: SIGTERM failed: {str(e)}")
    
    if proc.poll() is None:
        print(f"Process {proc.pid} still alive after SIGTERM; sending SIGKILL")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            print("Sent SIGKILL to process group")
            proc.wait(timeout=10)
            killed = True
        except ProcessLookupError:
            print("Process already terminated")
            return
        except Exception as e:
            print(f"Warning: SIGKILL failed: {str(e)}")
    
    if not killed and proc.poll() is None:
        print(f"ERROR: Process {proc.pid} could not be terminated - manual cleanup may be required")
        # Try to kill any remaining vllm processes on the port
        try:
            import subprocess
            subprocess.run(["pkill", "-f", "vllm.*serve"], check=False)
            print("Attempted to kill any remaining vLLM processes")
        except Exception as e:
            print(f"Warning: pkill failed: {str(e)}")
    else:
        print("vLLM process terminated")

def query_vllm_server(port, prompt, max_retries=3, retry_delay=1):
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "prompt": prompt
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            response_json = response.json()
            
            if not response_json.get("choices"):
                raise RuntimeError("Server returned empty choices array")
            
            return response_json["choices"][0]["text"]
            
        except requests.Timeout:
            last_error = "Request timed out"
        except requests.ConnectionError:
            last_error = "Failed to connect to server"
        except requests.HTTPError as e:
            last_error = f"Server returned HTTP {e.response.status_code}"
        except (KeyError, json.JSONDecodeError) as e:
            last_error = f"Invalid response format: {str(e)}"
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            
        if attempt < max_retries - 1:
            print(f"Attempt {attempt + 1} failed: {last_error}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
        
    raise RuntimeError(f"Failed to query vLLM server after {max_retries} attempts. Last error: {last_error}")

def cleanup_zombie_vllm_processes(ports_to_check=None):
    """Clean up any zombie vLLM processes that might be blocking ports."""
    print("Checking for zombie vLLM processes...")
    
    # Default to checking common ports if none specified
    if ports_to_check is None:
        ports_to_check = [8000]  # Legacy default
    
    try:
        # Check if any of the specified ports are in use
        ports_in_use = []
        for port in ports_to_check:
            if not check_port_available(port):
                ports_in_use.append(port)
        
        if not ports_in_use:
            print(f"All specified ports {ports_to_check} are available - no cleanup needed")
            return
            
        print(f"Ports {ports_in_use} are in use - attempting cleanup...")
        
        # Try to find and kill vLLM processes
        result = subprocess.run(
            ["pgrep", "-f", "vllm.*serve"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"Found {len(pids)} vLLM processes: {pids}")
            
            # Kill each process
            for pid in pids:
                if pid:
                    try:
                        print(f"Killing process {pid}...")
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(1)
                        # Force kill if still running
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # Process already terminated
                    except (ProcessLookupError, ValueError):
                        pass  # Process already terminated or invalid PID
                        
            # Wait a bit for processes to clean up
            time.sleep(3)
            
            # Check if ports are now available
            still_in_use = []
            for port in ports_in_use:
                if not check_port_available(port):
                    still_in_use.append(port)
            
            if not still_in_use:
                print(f"✓ All ports {ports_in_use} are now available after cleanup")
            else:
                print(f"⚠ Ports {still_in_use} are still in use after cleanup")
                
        else:
            print("No vLLM processes found")
            
    except Exception as e:
        print(f"Warning: Cleanup failed: {str(e)}")
        
    print("Cleanup complete") 