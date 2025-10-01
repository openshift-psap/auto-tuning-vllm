import json
import os
import subprocess
import time

from src.serving.utils import get_last_log_lines


def run_guidellm(guidellm_args, log_file):
    cmd = ["guidellm"] + guidellm_args
    print(f"Launching guidellm: {' '.join(cmd)}")

    try:
        with open(log_file, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"{'=' * 50}\n\n")
            f.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=f,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            print(f"guidellm launched successfully at {time.time()}")
    except FileNotFoundError:
        raise RuntimeError(
            "guidellm binary not found. Is guidellm installed and in PATH?"
        )
    except OSError as e:
        raise RuntimeError(
            f"Failed to start guidellm: {e.strerror}\nCommand: {' '.join(cmd)}"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error starting guidellm: {str(e)}")

    try:
        proc.wait(timeout=2400)
        print(f"guidellm process completed with return code: {proc.returncode}")
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("guidellm process timed out after 40 minutes")
    except Exception as e:
        proc.kill()
        raise RuntimeError(f"Error waiting for guidellm: {str(e)}")

    print(f"guidellm completed with return code: {proc.returncode} at {time.time()}")
    if proc.returncode != 0:
        last_logs = get_last_log_lines(log_file)
        raise RuntimeError(
            f"guidellm failed with code {proc.returncode}\n"
            f"Last log lines:\n{last_logs}\n"
            f"Check the full log file for details: {log_file}"
        )

    print(f"guidellm completed successfully at {time.time()}")

    return proc.returncode


def parse_benchmarks(bench_file):
    if not os.path.exists(bench_file):
        raise RuntimeError(f"Benchmark file missing: {bench_file}")

    try:
        with open(bench_file) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in benchmark file {bench_file}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error reading benchmark file {bench_file}: {str(e)}")

    try:
        stats = data["benchmarks"][0]
        metrics = stats["metrics"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"Invalid benchmark data structure in {bench_file}: {str(e)}"
        )

    REQUIRED_METRICS = [
        "requests_per_second",
        "request_concurrency",
        "request_latency",
        "prompt_token_count",
        "output_token_count",
        "time_to_first_token_ms",
        "time_per_output_token_ms",
        "inter_token_latency_ms",
        "output_tokens_per_second",
        "tokens_per_second",
    ]

    result = {}
    for metric in REQUIRED_METRICS:
        try:
            metric_data = metrics[metric]["successful"]

            result[metric] = metric_data["median"]

            percentiles = metric_data.get("percentiles", {})
            result[f"{metric}_p95"] = percentiles.get("p95")

            result[f"{metric}_p50"] = percentiles.get("p50", metric_data["median"])
            result[f"{metric}_p90"] = percentiles.get("p90")
            result[f"{metric}_p99"] = percentiles.get("p99")

        except KeyError:
            raise RuntimeError(f"Missing required metric in {bench_file}: {metric}")

    return result
