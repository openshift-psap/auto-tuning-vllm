import os
import sys
import glob
import re
import optuna
import yaml
import argparse
from optuna.samplers import TPESampler, RandomSampler, NSGAIISampler, GridSampler
from optuna.integration import BoTorchSampler
from src.serving.utils import validate_huggingface_model, save_config_to_study
from src.serving.optimization import run_parallel_trials, p95_latency_objective_function_DEPRECATED
from src.serving.run_baseline import run_baseline_test
from src.serving.vllm_server import cleanup_zombie_vllm_processes

SRC_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
STUDIES_ROOT = os.path.join(PROJECT_DIR, "studies")
os.makedirs(STUDIES_ROOT, exist_ok=True)

# Load vLLM configuration
VLLM_CONFIG_PATH = os.path.join(PROJECT_DIR, "vllm_config.yaml")

try:
    with open(VLLM_CONFIG_PATH, 'r') as f:
        vllm_config = yaml.safe_load(f)
except FileNotFoundError:
    sys.exit(f"Config not found at {VLLM_CONFIG_PATH}.")
except yaml.YAMLError as e:
    sys.exit(f"Config syntax error in {VLLM_CONFIG_PATH}: {e}")

study_dirs = glob.glob(os.path.join(STUDIES_ROOT, "study_*"))
study_ids = []
for d in study_dirs:
    m = re.match(r".*study_(\d+)$", d)
    if m:
        study_ids.append(int(m.group(1)))

STUDY_ID = max(study_ids) + 1 if study_ids else 1
STUDY_DIR = os.path.join(STUDIES_ROOT, f"study_{STUDY_ID}")
VLLM_LOGS_DIR = os.path.join(STUDY_DIR, "vllm_logs")
GUIDELLM_LOGS_DIR = os.path.join(STUDY_DIR, "guidellm_logs")

os.makedirs(STUDY_DIR, exist_ok=True)
os.makedirs(VLLM_LOGS_DIR, exist_ok=True)
os.makedirs(GUIDELLM_LOGS_DIR, exist_ok=True)

# Save configuration file to study directory
save_config_to_study(VLLM_CONFIG_PATH, STUDY_DIR, STUDY_ID)

print(f"Logging this study's data to: {STUDY_DIR}")
print(f"Logging vLLM server logs to: {VLLM_LOGS_DIR}")
print(f"Logging guidellm logs to: {GUIDELLM_LOGS_DIR}")


def build_grid_search_space(config):
    """Build search space for GridSampler from vLLM config"""
    search_space = {}
    parameters = config.get("parameters", {})
    
    total_combinations = 1
    
    print("DEBUG: Building grid search space from config...")
    
    for param_name, param_config in parameters.items():
        if not param_config.get("enabled", False):
            print(f"DEBUG: Skipping disabled parameter: {param_name}")
            continue
        
        param_key = param_name.replace("-", "_")
        print(f"DEBUG: Processing parameter: {param_name} -> {param_key}")
        
        if "options" in param_config:
            # Discrete options
            values = param_config["options"]
            search_space[param_key] = values
            total_combinations *= len(values)
            print(f"DEBUG: Options parameter {param_key}: {len(values)} values {values}")
            
        elif "range" in param_config:
            # Range parameters - convert to discrete list
            range_config = param_config["range"]
            start = range_config["start"]
            end = range_config["end"]
            step = range_config["step"]
            
            # Generate all values in range
            values = []
            current = start
            while current <= end:
                values.append(current)
                current += step
                # Handle floating point precision issues
                current = round(current, 10)
            
            search_space[param_key] = values
            total_combinations *= len(values)
            print(f"DEBUG: Range parameter {param_key}: {len(values)} values from {start} to {end} step {step}")
            
        elif "level" in param_config:
            # Level parameters (like compilation_config)
            # These are treated as categorical choices, not ranges
            levels = param_config["level"]
            if isinstance(levels, list):
                search_space[param_key] = levels
                total_combinations *= len(levels)
                print(f"DEBUG: Level parameter {param_key}: {len(levels)} values {levels}")
        else:
            print(f"WARNING: Unknown parameter type for {param_name}: {param_config}")
    
    print(f"DEBUG: Final search space: {search_space}")
    print(f"DEBUG: Total combinations: {total_combinations}")
    
    return search_space, total_combinations



def main():
    parser = argparse.ArgumentParser(description='vLLM Performance Optimization')
    parser.add_argument('--mode', choices=['config', 'p95_latency'], default='config',
                        help='Optimization mode: config (use vllm_config.yaml) or p95_latency (minimize p95 latency)')
    parser.add_argument('--model', type=str, 
                        help='HuggingFace model name (default: Qwen/Qwen3-32B-FP8)')
    parser.add_argument('--max-seconds', type=int, default=240,
                        help='Duration for each trial in seconds (default: 240)')
    parser.add_argument('--prompt-tokens', type=int, default=1000,
                        help='Number of prompt tokens for synthetic data (default: 1000)')
    parser.add_argument('--output-tokens', type=int, default=1000,
                        help='Number of output tokens for synthetic data (default: 1000)')
    parser.add_argument('--n-trials', type=int, 
                        help='Number of optimization trials (overrides config)')
    parser.add_argument('--dataset', type=str,
                        help='Dataset for guidellm: HuggingFace dataset ID, local path to dataset file (CSV, JSONL, etc.), or leave empty to use synthetic data')
    parser.add_argument('--gpus', type=str, default="0",
                        help='Comma-separated list of GPU IDs to use (default: "0" for single GPU)')
    parser.add_argument('--baseline-gpu', type=int, default=0,
                        help='GPU ID to use for baseline test (default: 0)')
    parser.add_argument('--start-port', type=int, default=60000,
                        help='Starting port number for parallel trials (default: 60000)')
    
    args = parser.parse_args()
    
    model = args.model if args.model else "Qwen/Qwen3-32B-FP8"
    
    if not validate_huggingface_model(model):
        print(f"Error: Invalid HuggingFace model: {model}")
        sys.exit(1)
    
    # Clean up any zombie vLLM processes before starting
    cleanup_zombie_vllm_processes()
    
    print("=" * 80)
    print("RUNNING BASELINE TESTS WITH DIFFERENT CONCURRENCY LEVELS")
    print("=" * 80)
    print(f"Selected GPU for baseline: {args.baseline_gpu}")
    
    # Run 5 baseline tests with concurrency levels from 50 to 250 in steps of 50
    concurrency_levels = [50] #, 100, 150, 200
    baseline_results = []
    
    for concurrency in concurrency_levels:
        print(f"\n{'='*40}")
        print(f"BASELINE TEST: Concurrency {concurrency}")
        print(f"{'='*40}")
        
        baseline_metrics = run_baseline_test(
            model, args.max_seconds, args.prompt_tokens, args.output_tokens, args.dataset,
            STUDY_DIR, VLLM_LOGS_DIR, GUIDELLM_LOGS_DIR, STUDY_ID, concurrency, gpu_id=args.baseline_gpu
        )
        
        if baseline_metrics is not None:
            baseline_results.append({
                'concurrency': concurrency,
                'metrics': baseline_metrics
            })
            print(f"Concurrency {concurrency} - Performance: {baseline_metrics['output_tokens_per_second']:.2f} output tokens/second")
            print(f"Concurrency {concurrency} - Latency: {baseline_metrics['request_latency']:.2f} ms")
        else:
            print(f"Baseline test failed for concurrency {concurrency}")
    print("=" * 80)
    
    # Use the first successful baseline for optimization comparison (or None if all failed)
    baseline_metrics = baseline_results[0]['metrics'] if baseline_results else None
    
    db_path = os.path.join(STUDY_DIR, "optuna.db")
    storage_url = f"sqlite:///{db_path}"
    
    if args.mode == 'p95_latency':
        print("=== P95 LATENCY OPTIMIZATION MODE ===")
        print(f"Model: {model}")
        print(f"Trial duration: {args.max_seconds} seconds")
        if args.dataset:
            print(f"Dataset: {args.dataset}")
        else:
            print(f"Prompt tokens: {args.prompt_tokens}")
            print(f"Output tokens: {args.output_tokens}")
        n_trials = args.n_trials if args.n_trials else 100
        print(f"Number of trials: {n_trials}")
        print("Objective: Minimize P95 end-to-end latency")
        print("=" * 50)
        
        study = optuna.create_study(
            storage=storage_url,
            study_name=f"vllm_p95_latency_run{STUDY_ID}",
            direction="minimize",
            load_if_exists=True
        )
        
        def objective_function(trial):
            return p95_latency_objective_function_DEPRECATED(
                trial, model, args.max_seconds, args.prompt_tokens, args.output_tokens, args.dataset,
                vllm_config, STUDY_DIR, VLLM_LOGS_DIR, GUIDELLM_LOGS_DIR, STUDY_ID
            )
        
        print(f"Starting P95 latency optimization with {n_trials} trials...")
        study.optimize(objective_function, n_trials=n_trials)
        
        print("\n" + "="*80)
        print("P95 LATENCY OPTIMIZATION RESULTS")
        print("="*80)
        
        best_trial = study.best_trial
        best_p95_latency = best_trial.value
        
        print(f"Best P95 latency achieved: {best_p95_latency:.2f} ms")
        if baseline_metrics and baseline_metrics.get('request_latency_p95'):
            baseline_p95 = baseline_metrics['request_latency_p95']
            improvement = ((baseline_p95 - best_p95_latency) / baseline_p95) * 100
            print(f"P95 latency improvement over baseline: {improvement:.2f}%")
        
        print("Best trial parameters:")
        for param, value in best_trial.params.items():
            print(f"  --{param.replace('_', '-')}: {value}")
        
        if hasattr(best_trial, 'user_attrs'):
            print("\nAdditional metrics from best trial:")
            for attr, value in best_trial.user_attrs.items():
                if 'latency' in attr.lower() and value is not None:
                    print(f"  {attr}: {value}")
        
        print("\nTo use this configuration:")
        print(f"vllm serve {model} \\")
        for param, value in best_trial.params.items():
            print(f"  --{param.replace('_', '-')} {value} \\")
        print("  --port 8000")
        
        return
    
    optimization_config = vllm_config.get("optimization", {})
    optimization_approach = optimization_config.get("approach", "single_objective")
    
    print("=== CONFIG-BASED OPTIMIZATION MODE ===")
    print(f"Approach: {optimization_approach}")
    print(f"Model: {model}")
    print(f"Trial duration: {args.max_seconds} seconds")
    if args.dataset:
        print(f"Dataset: {args.dataset}")
    else:
        print(f"Prompt tokens: {args.prompt_tokens}")
        print(f"Output tokens: {args.output_tokens}")
    n_trials = args.n_trials if args.n_trials else optimization_config.get("n_trials", 200)
    print(f"Number of trials: {n_trials}")
    print("=" * 50)
    
    sampler_name = optimization_config.get("sampler", "tpe")
    
    if sampler_name == "botorch":
        sampler = BoTorchSampler(
            n_startup_trials=20
        )

    elif sampler_name == "nsga2":
        sampler = NSGAIISampler()
    elif sampler_name == "tpe":
        sampler = TPESampler()
    elif sampler_name == "random":
        sampler = RandomSampler()
    elif sampler_name == "grid":
        search_space, total_combinations = build_grid_search_space(vllm_config)
        
        print("\n=== GRID SAMPLER CONFIGURATION ===")
        print("Parameters to optimize:")
        for param, values in search_space.items():
            print(f"  {param}: {len(values)} values {values}")
        print(f"Total possible combinations: {total_combinations:,}")
        print(f"GridSampler will test all {total_combinations:,} combinations")
        
        # Set seed for reproducible results
        sampler = GridSampler(search_space=search_space, seed=42)
    else:
        sampler = TPESampler()
    
    if optimization_approach == "multi_objective":
        study = optuna.create_study(
            storage=storage_url,
            study_name=f"vllm_tuning_run{STUDY_ID}_multi",
            directions=["maximize", "minimize"],
            sampler=sampler,
            load_if_exists=True
        )
        

        
        print(f"Using multi-objective optimization (throughput vs latency) with {sampler_name} sampler")
        
    else:
        study = optuna.create_study(
            storage=storage_url,
            study_name=f"vllm_tuning_run{STUDY_ID}_single",
            direction="maximize",
            sampler=sampler,
            load_if_exists=True
        )
        

        
        print("Using single-objective optimization (maximize throughput)")
        print("Result: Highest throughput configuration (latency not considered)")

    print(f"\nStarting Optuna trials with {optimization_approach} approach...")
    
    # Parse GPU IDs and always use parallel workflow
    gpu_ids = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    print(f"Using dynamic GPU scheduling with {sampler_name} sampler on GPUs: {gpu_ids}")
    
    # Check ALL required ports are available before starting study
    from src.serving.utils import check_all_ports_available_for_study
    
    all_available, unavailable_ports, required_ports = check_all_ports_available_for_study(gpu_ids, args.start_port)
    print(f"\nChecking port availability for {len(gpu_ids)} GPU(s) starting at port {args.start_port}. Required ports: {required_ports}")
    
    if not all_available:
        # NOISY ERROR - exactly as specified
        print("\n" + "="*80)
        print("FATAL ERROR: PORT CONFLICT DETECTED")
        print("="*80)
        print(f"The following ports are already in use: {unavailable_ports}")
        print(f"Required ports for GPU trials: {required_ports}")
        print("")
        print("STUDY CANNOT START WITH PORT CONFLICTS.")
        print("")
        print("Please:")
        print("1. Kill any processes using these ports, OR")
        print("2. Use a different --start-port value")
        print("="*80)
        sys.exit(1)
    
    print(f"All {len(required_ports)} required ports are available")
    
    # Always use the parallel workflow (works with 1 or more GPUs)
    study = run_parallel_trials(
        study, model, args.max_seconds, args.prompt_tokens, args.output_tokens, args.dataset,
        vllm_config, STUDY_DIR, VLLM_LOGS_DIR, GUIDELLM_LOGS_DIR, STUDY_ID, gpu_ids, n_trials, args.start_port
    )

    print("\nOptimization Results:")
    if optimization_approach == "multi_objective":
        print("Multi-objective results:")
        print(f"Number of Pareto-optimal solutions: {len(study.best_trials)}")
        print("\nTop solutions:")
        for i, trial in enumerate(study.best_trials[:5]):
            throughput, latency = trial.values
            print(f"    Parameters: {trial.params}")
    
        
    else:
        print(f"Best trial value: {study.best_trial.value}")
        print(f"Best trial parameters: {study.best_trial.params}")
        if baseline_metrics is not None:
            improvement = ((study.best_trial.value - baseline_metrics["output_tokens_per_second"]) / baseline_metrics["output_tokens_per_second"]) * 100
            print(f"Improvement over baseline: {improvement:.2f}%")

if __name__ == "__main__":
    main()