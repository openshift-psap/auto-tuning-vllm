import os
import sys
from src.serving.vllm_server import build_vllm_command, start_vllm_server, stop_vllm_server
from src.serving.benchmarking import run_guidellm, parse_benchmarks

def run_baseline_test(model=None, max_seconds=None, prompt_tokens=None, output_tokens=None, dataset=None,
                     study_dir=None, vllm_logs_dir=None, guidellm_logs_dir=None, study_id=None, concurrency=50, gpu_id=0):
    # Use prescribed port range (default start port for baseline)
    from src.serving.utils import get_port_for_gpu
    port = get_port_for_gpu(gpu_id, start_port=60000)  # Default for baseline tests
    
    if model is None:
        model = "Qwen/Qwen3-32B-FP8"
    if max_seconds is None:
        max_seconds = 240
    if prompt_tokens is None:
        prompt_tokens = 1000
    if output_tokens is None:
        output_tokens = 1000
    
    vllm_log_file = os.path.join(vllm_logs_dir, f"vllm_server_logs_{study_id}.baseline.concurrency_{concurrency}.log")
    guidellm_log_file = os.path.join(guidellm_logs_dir, f"guidellm_logs_{study_id}.baseline.concurrency_{concurrency}.log")
    
    print(f"\nRunning baseline test with concurrency {concurrency}")
    print(f"Model: {model}")
    print(f"Duration: {max_seconds} seconds")
    print(f"Prompt tokens: {prompt_tokens}, Output tokens: {output_tokens}")
    print(f"Concurrency: {concurrency}")
    print(f"GPU ID: {gpu_id}")
    print(f"vLLM log file: {vllm_log_file}")
    print(f"guidellm log file: {guidellm_log_file}")
    
    # Set CUDA_VISIBLE_DEVICES for this specific GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Add memory management environment variables
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Add baseline-specific flags to manage memory usage
    baseline_flags = [

        "--max-num-seqs", str(concurrency),   # Limit concurrent sequencesoverhead
    ]
    
    vllm_cmd = build_vllm_command(model_name=model, port=port, candidate_flags=baseline_flags)
    vllm_proc = start_vllm_server(vllm_cmd, log_file=vllm_log_file, env=env)

    bench_file = os.path.join(study_dir, f"benchmarks_{study_id}.baseline.concurrency_{concurrency}.json")

    try:
        print("Starting guidellm benchmark for baseline...")
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
            "--max-seconds", str(max_seconds),
            "--output-path", bench_file
        ])

        run_guidellm(guidellm_args, guidellm_log_file)
        print("Baseline guidellm benchmark completed successfully")

        metrics = parse_benchmarks(bench_file)
        return metrics

    except Exception as e:
        print("Error during baseline test:", str(e), file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

    finally:
        stop_vllm_server(vllm_proc)
        print("Baseline vLLM server stopped.") 