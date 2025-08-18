import optuna
import sys
import time
import os
import threading
import concurrent.futures
from src.serving.vllm_server import build_vllm_command, start_vllm_server, stop_vllm_server
from src.serving.benchmarking import run_guidellm, parse_benchmarks

def generate_vllm_parameters(trial, config):
    candidate_flags = []
    
    parameters_config = config.get("parameters", {})
    
    for param_name, param_config in parameters_config.items():
        if not param_config.get("enabled", False):
            continue
            
        # Skip guidellm_concurrency as it's not a vLLM server parameter
        if param_name == "guidellm_concurrency":
            continue
            
        flag_name = param_config.get("name", param_name.replace("_", "-"))
        
        if "options" in param_config:
            value = trial.suggest_categorical(param_name, param_config["options"])
            candidate_flags.extend([f"--{flag_name}", str(value)])
            
        elif "level" in param_config:
            value = trial.suggest_categorical(param_name, param_config["level"])
            candidate_flags.extend([f"--{flag_name}", str(value)])
            
        elif "range" in param_config:
            range_config = param_config["range"]
            start = range_config["start"]
            end = range_config["end"]
            step = range_config.get("step", 1)
            
            if isinstance(step, float) or isinstance(start, float) or isinstance(end, float):
                value = trial.suggest_float(param_name, start, end, step=step)
            else:
                value = trial.suggest_int(param_name, start, end, step=step)
                
            candidate_flags.extend([f"--{flag_name}", str(value)])
    
    return candidate_flags

# DEPRECATED: This function is no longer used - all workflows now use run_parallel_trials
def run_single_trial_DEPRECATED(trial, model=None, max_seconds=None, prompt_tokens=None, output_tokens=None, dataset=None, 
                    vllm_config=None, study_dir=None, vllm_logs_dir=None, guidellm_logs_dir=None, study_id=None):
    # Use prescribed port range (GPU 0 for single trials)
    from src.serving.utils import get_port_for_gpu
    port = get_port_for_gpu(0, start_port=60000)  # Default for sequential trials
    
    if model is None:
        model = "Qwen/Qwen3-32B-FP8"
    if max_seconds is None:
        max_seconds = 240
    if prompt_tokens is None:
        prompt_tokens = 1000
    if output_tokens is None:
        output_tokens = 1000

    candidate_flags = generate_vllm_parameters(trial, vllm_config)
    
    # Get guidellm concurrency from config or use default
    parameters_config = vllm_config.get("parameters", {})
    concurrency_config = parameters_config.get("guidellm_concurrency", {})
    if concurrency_config.get("enabled", False):
        concurrency = trial.suggest_int("guidellm_concurrency", 
                                       concurrency_config["range"]["start"],
                                       concurrency_config["range"]["end"],
                                       step=concurrency_config["range"]["step"])
    else:
        concurrency = 50  # Default fallback
    
    candidate_flags.extend(["--max-num-seqs", str(concurrency)])

    trial_id = trial.number
    vllm_log_file = os.path.join(vllm_logs_dir, f"vllm_server_logs_{study_id}.{trial_id}.log")
    guidellm_log_file = os.path.join(guidellm_logs_dir, f"guidellm_logs_{study_id}.{trial_id}.log")
    
    print(f"\nStarting trial {trial_id}")
    print(f"Model: {model}")
    print(f"Duration: {max_seconds} seconds")
    print(f"Prompt tokens: {prompt_tokens}, Output tokens: {output_tokens}")
    print(f"Concurrency: {concurrency}")
    print(f"Parameters: {' '.join(candidate_flags)}")
    print(f"vLLM log file: {vllm_log_file}")
    print(f"guidellm log file: {guidellm_log_file}")

    vllm_cmd = build_vllm_command(model_name=model, port=port, candidate_flags=candidate_flags)
    vllm_proc = start_vllm_server(vllm_cmd, log_file=vllm_log_file)

    bench_file = os.path.join(
        study_dir,
        f"benchmarks_{study_id}.{trial_id}.json"
    )

    try:
        print("Starting guidellm benchmark...")
        guidellm_args = [ # TODO: try setting it to have a set RPS
            "benchmark",
            "--target",      f"http://localhost:{port}",
            "--model",       model,
            "--processor",   model,
        ]
        
        if dataset:
            print(f"Using custom dataset: {dataset}")
            guidellm_args.extend(["--data", dataset])
        else:
            print(f"Using synthetic data: {prompt_tokens} prompt tokens, {output_tokens} output tokens")
            guidellm_args.extend(["--data", f'{{"prompt_tokens":{prompt_tokens},"output_tokens":{output_tokens}}}'])
        
        guidellm_args.extend([
            "--rate-type",   "concurrent",
            "--rate",        str(concurrency),
            "--warmup-percent", "0.1", # Add warmup
            "--max-seconds", str(max_seconds),
            "--output-path", bench_file
        ])

        run_guidellm(guidellm_args, guidellm_log_file)
        print("guidellm benchmark completed successfully")

        metrics = parse_benchmarks(bench_file)
        
        for metric_name, value in metrics.items():
            if value is not None:
                trial.set_user_attr(metric_name, value)

        return metrics

    finally:
        stop_vllm_server(vllm_proc)
        print("vLLM server stopped.")
        interval = vllm_config["settings"].get("trial_interval", 30)
        print(f"Waiting {interval} seconds before next trial...")
        time.sleep(interval)

# DEPRECATED: This function is no longer used - all workflows now use run_parallel_trials  
def multi_objective_function_DEPRECATED(trial, model=None, max_seconds=None, prompt_tokens=None, output_tokens=None, dataset=None,
                            vllm_config=None, study_dir=None, vllm_logs_dir=None, guidellm_logs_dir=None, study_id=None):
    try:
        metrics = run_single_trial_DEPRECATED(trial, model, max_seconds, prompt_tokens, output_tokens, dataset,
                                 vllm_config, study_dir, vllm_logs_dir, guidellm_logs_dir, study_id)
        
        throughput = float(metrics["output_tokens_per_second"])
        latency = float(metrics["request_latency"])
        
        return throughput, latency
        
    except Exception as e:
        print(f"Error during multi-objective trial {trial.number}:", str(e), file=sys.stderr)
        import traceback
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        trial.set_user_attr("traceback", traceback.format_exc())
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")

# DEPRECATED: This function is no longer used - all workflows now use run_parallel_trials
def objective_DEPRECATED(trial, model=None, max_seconds=None, prompt_tokens=None, output_tokens=None, dataset=None,
             vllm_config=None, study_dir=None, vllm_logs_dir=None, guidellm_logs_dir=None, study_id=None):
    try:
        metrics = run_single_trial_DEPRECATED(trial, model, max_seconds, prompt_tokens, output_tokens, dataset,
                                 vllm_config, study_dir, vllm_logs_dir, guidellm_logs_dir, study_id)
        print(f"DEBUG: Trial {trial.number}, Metric value to return: {metrics['output_tokens_per_second']}")
        
        return float(metrics["output_tokens_per_second"])

    except Exception as e:
        print(f"Error during trial {trial.number}:", str(e), file=sys.stderr)
        import traceback
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        trial.set_user_attr("traceback", traceback.format_exc())
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")

# DEPRECATED: This function is no longer used - all workflows now use run_parallel_trials
def p95_latency_objective_function_DEPRECATED(trial, model, max_seconds, prompt_tokens, output_tokens, dataset=None,
                                  vllm_config=None, study_dir=None, vllm_logs_dir=None, guidellm_logs_dir=None, study_id=None):
    try:
        metrics = run_single_trial_DEPRECATED(trial, model, max_seconds, prompt_tokens, output_tokens, dataset,
                                 vllm_config, study_dir, vllm_logs_dir, guidellm_logs_dir, study_id)
        
        p95_latency = metrics.get("request_latency_p95")
        
        if p95_latency is None:
            p95_latency = metrics["request_latency"]
            print(f"Warning: p95 latency not available, using median: {p95_latency} ms")
        
        for metric_name, value in metrics.items():
            if value is not None:
                trial.set_user_attr(metric_name, value)
        
        latency_value = float(p95_latency)
        
        print(f"Trial {trial.number}: P95 latency = {latency_value:.2f} ms")
        
        return latency_value
        
    except Exception as e:
        print(f"Error during p95 latency trial {trial.number}:", str(e), file=sys.stderr)
        import traceback
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        trial.set_user_attr("traceback", traceback.format_exc())
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")

def run_single_trial_parallel(trial, model, max_seconds, prompt_tokens, output_tokens, dataset,
                              vllm_config, study_dir, vllm_logs_dir, guidellm_logs_dir, study_id, gpu_id, start_port):
    """
    Run a single trial on a specific GPU for parallel optimization
    """
    # Use prescribed port range for parallel trials
    from src.serving.utils import get_port_for_gpu
    port = get_port_for_gpu(gpu_id, start_port)
    
    if model is None:
        model = "Qwen/Qwen3-32B-FP8"
    if max_seconds is None:
        max_seconds = 240
    if prompt_tokens is None:
        prompt_tokens = 1000
    if output_tokens is None:
        output_tokens = 1000

    candidate_flags = generate_vllm_parameters(trial, vllm_config)
    
    # Get guidellm concurrency from config or use default
    parameters_config = vllm_config.get("parameters", {})
    concurrency_config = parameters_config.get("guidellm_concurrency", {})
    if concurrency_config.get("enabled", False):
        concurrency = trial.suggest_int("guidellm_concurrency", 
                                       concurrency_config["range"]["start"],
                                       concurrency_config["range"]["end"],
                                       step=concurrency_config["range"]["step"])
    else:
        concurrency = 50  # Default fallback
    
    candidate_flags.extend(["--max-num-seqs", str(concurrency)])
    
    # Add GPU specification
    candidate_flags.extend(["--tensor-parallel-size", "1"])
    
    # Set CUDA_VISIBLE_DEVICES for this specific GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    trial_id = trial.number
    vllm_log_file = os.path.join(vllm_logs_dir, f"vllm_server_logs_{study_id}.{trial_id}_gpu{gpu_id}.log")
    guidellm_log_file = os.path.join(guidellm_logs_dir, f"guidellm_logs_{study_id}.{trial_id}_gpu{gpu_id}.log")
    
    print(f"\nStarting trial {trial_id} on GPU {gpu_id}")
    print(f"Model: {model}")
    print(f"Duration: {max_seconds} seconds")
    print(f"Prompt tokens: {prompt_tokens}, Output tokens: {output_tokens}")
    print(f"Concurrency: {concurrency}")
    print(f"Port: {port}")
    print(f"Parameters: {' '.join(candidate_flags)}")
    print(f"vLLM log file: {vllm_log_file}")
    print(f"guidellm log file: {guidellm_log_file}")

    vllm_cmd = build_vllm_command(model_name=model, port=port, candidate_flags=candidate_flags)
    vllm_proc = start_vllm_server(vllm_cmd, log_file=vllm_log_file, env=env)

    bench_file = os.path.join(
        study_dir,
        f"benchmarks_{study_id}.{trial_id}_gpu{gpu_id}.json"
    )

    try:
        print(f"Starting guidellm benchmark on GPU {gpu_id}...")
        guidellm_args = [
            "benchmark",
            "--target",      f"http://localhost:{port}",
            "--model",       model,
            "--processor",   model,
        ]
        
        if dataset:
            print(f"Using custom dataset: {dataset}")
            guidellm_args.extend(["--data", dataset])
        else:
            print(f"Using synthetic data: {prompt_tokens} prompt tokens, {output_tokens} output tokens")
            guidellm_args.extend(["--data", f'{{"prompt_tokens":{prompt_tokens},"output_tokens":{output_tokens}}}'])
        
        guidellm_args.extend([
            "--rate-type",   "concurrent",
            "--rate",        str(concurrency),
            "--warmup-percent", "0.1",
            "--max-seconds", str(max_seconds),
            "--output-path", bench_file
        ])

        run_guidellm(guidellm_args, guidellm_log_file)
        print(f"guidellm benchmark completed successfully on GPU {gpu_id}")

        metrics = parse_benchmarks(bench_file)
        
        for metric_name, value in metrics.items():
            if value is not None:
                trial.set_user_attr(metric_name, value)

        return metrics

    finally:
        stop_vllm_server(vllm_proc)
        print(f"vLLM server stopped on GPU {gpu_id}.")
        interval = vllm_config["settings"].get("trial_interval", 10)  # Shorter interval for parallel
        print(f"Waiting {interval} seconds before next trial on GPU {gpu_id}...")
        time.sleep(interval)


def run_parallel_trials(study, model, max_seconds, prompt_tokens, output_tokens, dataset,
                       vllm_config, study_dir, vllm_logs_dir, guidellm_logs_dir, study_id, gpu_ids, n_trials, start_port):
    """
    Run optimization trials in parallel across multiple GPUs with dynamic scheduling.
    When any GPU completes (success or failure), immediately start the next trial.
    
    Args:
        study: Optuna study object
        model: HuggingFace model name
        max_seconds: Duration for each trial
        prompt_tokens: Number of prompt tokens
        output_tokens: Number of output tokens  
        dataset: Dataset path
        vllm_config: vLLM configuration dict
        study_dir: Study directory path
        vllm_logs_dir: vLLM logs directory
        guidellm_logs_dir: Guidellm logs directory
        study_id: Study ID
        gpu_ids: List of GPU IDs to use for parallel trials
        n_trials: Total number of trials to run
        
    Returns:
        Updated study object
    """
    print("\nParallel optimization")
    print(f"GPUs: {gpu_ids}")
    print(f"Total trials: {n_trials}")
    print(f"Dynamic workers: {len(gpu_ids)}")
    print("Note: Next trial starts immediately when any GPU becomes available")
    print("=" * 60)
    
    # GPU state tracking - use prescribed port range
    from src.serving.utils import get_port_for_gpu
    gpu_states = {gpu_id: {'status': 'idle', 'trial_id': None, 'trial_obj': None, 'port': get_port_for_gpu(gpu_id, start_port), 'future': None} 
                  for gpu_id in gpu_ids}
    
    completed_trials = 0
    failed_trials = 0
    
    # Thread-safe counters
    stats_lock = threading.Lock()
    
    def parallel_objective_wrapper(gpu_id, trial):
        """Objective function wrapper for parallel trials - supports both single and multi-objective"""
        nonlocal completed_trials, failed_trials
        print(f"\n[GPU {gpu_id}] Starting trial {trial.number}")
        
        try:
            metrics = run_single_trial_parallel(
                trial, model, max_seconds, prompt_tokens, output_tokens, dataset,
                vllm_config, study_dir, vllm_logs_dir, guidellm_logs_dir, study_id, gpu_id, start_port
            )
            
            tokens_per_second = float(metrics["output_tokens_per_second"])
            
            # Determine return format based on study type
            if hasattr(study, 'directions') and len(study.directions) > 1:
                # Multi-objective study - return (throughput, latency)
                latency = float(metrics["request_latency"]) 
                print(f"[GPU {gpu_id}] Trial {trial.number}: {tokens_per_second:.2f} tokens/s, {latency:.2f}ms latency")
                
                with stats_lock:
                    completed_trials += 1
                    
                return tokens_per_second, latency
            else:
                # Single-objective study - return just throughput
                print(f"[GPU {gpu_id}] Trial {trial.number}: {tokens_per_second:.2f} tokens/s")
                
                with stats_lock:
                    completed_trials += 1
                    
                return tokens_per_second
                
        except Exception as e:
            error_msg = str(e)
            print(f"[GPU {gpu_id}] Trial {trial.number} failed: {error_msg}")
            
            # Check if it's an OOM error
            if "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
                print(f"[GPU {gpu_id}] OOM detected - GPU memory exhausted, moving to next trial")
            
            # Match sequential objective function error handling exactly
            import traceback
            trial.set_user_attr("error", str(e))
            trial.set_user_attr("traceback", traceback.format_exc())
            
            with stats_lock:
                failed_trials += 1
                
            # Use proper Optuna exception like sequential trials
            raise optuna.TrialPruned(f"Trial failed: {str(e)}")
    
    # Use ThreadPoolExecutor for dynamic scheduling
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        
        while completed_trials + failed_trials < n_trials:
            # Check for completed GPU tasks and start new ones
            for gpu_id in gpu_ids:
                gpu_state = gpu_states[gpu_id]
                
                # If GPU is running a trial, check if it's done
                if gpu_state['status'] == 'running' and gpu_state['future']:
                    if gpu_state['future'].done():
                        try:
                            result = gpu_state['future'].result()
                            # Report the result back to Optuna
                            trial = gpu_state['trial_obj']
                            study.tell(trial, result)
                            print(f"[GPU {gpu_id}] Trial {gpu_state['trial_id']} completed successfully, value: {result:.2f}")
                        except Exception as e:
                            print(f"[GPU {gpu_id}] Trial {gpu_state['trial_id']} failed: {e}")
                            # Mark trial as failed in Optuna
                            if 'trial_obj' in gpu_state and gpu_state['trial_obj']:
                                try:
                                    study.tell(gpu_state['trial_obj'], state=optuna.trial.TrialState.FAIL)
                                except Exception:
                                    pass  # Don't fail if we can't mark it as failed
                        
                        gpu_state['status'] = 'idle'
                        gpu_state['future'] = None
                        gpu_state['trial_id'] = None
                        gpu_state['trial_obj'] = None
                
                # If GPU is idle and we have more trials to run, start the next one
                if (gpu_state['status'] == 'idle' and 
                    completed_trials + failed_trials < n_trials):
                    
                    try:
                        trial = study.ask()
                        
                        # Submit trial to this GPU using proper objective wrapper
                        future = executor.submit(parallel_objective_wrapper, gpu_id, trial)
                        gpu_state['future'] = future
                        gpu_state['status'] = 'running'
                        gpu_state['trial_id'] = trial.number
                        gpu_state['trial_obj'] = trial
                        
                        print(f"[GPU {gpu_id}] Assigned trial {trial.number} (Progress: {completed_trials + failed_trials}/{n_trials})")
                        
                    except Exception as e:
                        print(f"Error getting next trial: {e}")
                        break
            
            # Brief sleep to avoid busy waiting
            time.sleep(0.5)
        
        # Wait for any remaining trials to complete and report results
        print("Waiting for remaining trials to complete...")
        for gpu_id in gpu_ids:
            if gpu_states[gpu_id]['future']:
                try:
                    result = gpu_states[gpu_id]['future'].result()
                    # Report the result back to Optuna
                    if gpu_states[gpu_id]['trial_obj']:
                        study.tell(gpu_states[gpu_id]['trial_obj'], result)
                        print(f"[GPU {gpu_id}] Final trial {gpu_states[gpu_id]['trial_id']} completed, value: {result:.2f}")
                except Exception as e:
                    print(f"Final trial on GPU {gpu_id} failed: {e}")
                    # Mark trial as failed in Optuna
                    if gpu_states[gpu_id]['trial_obj']:
                        try:
                            study.tell(gpu_states[gpu_id]['trial_obj'], state=optuna.trial.TrialState.FAIL)
                        except Exception:
                            pass
    
    print("\nParallel optimization complete!")
    print(f"Completed trials: {completed_trials}")
    print(f"Failed trials: {failed_trials}")
    print(f"Total trials: {completed_trials + failed_trials}")
    
    # Report results based on study type
    if len(study.trials) > 0:
        try:
            # Try single-objective approach first
            best_trial = study.best_trial
            print(f"Best trial: {best_trial.value:.2f} tokens/s")
            print(f"Best parameters: {best_trial.params}")
        except RuntimeError:
            # Multi-objective study - show Pareto front
            print("Multi-objective results:")
            print(f"Number of Pareto-optimal solutions: {len(study.best_trials)}")
            print("Top solutions:")
            for i, trial in enumerate(study.best_trials[:5]):
                throughput, latency = trial.values
                print(f"  Solution {i+1}: Throughput={throughput:.2f} tokens/s, Latency={latency:.2f}ms")
                print(f"    Parameters: {trial.params}")
    else:
        print("No successful trials completed.")
    
    return study


def generate_command_line(trial_params):
    args = []
    for param, value in trial_params.items():
        arg_name = "--" + param.replace("_", "-")
        args.append(f"{arg_name} {value}")
    
    return " ".join(args) 