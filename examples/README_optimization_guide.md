# Optimization Configuration Guide

This guide explains how to configure optimization objectives for auto-tune-vllm to get the best results for your specific use case.

## Quick Start

### Simple Presets (Recommended for Beginners)

```yaml
# High throughput (maximize tokens/second)
optimization:
  preset: "high_throughput"
  n_trials: 100

# Low latency (minimize response time)
optimization:
  preset: "low_latency"
  n_trials: 100

# Balanced (find throughput vs latency trade-offs)
optimization:
  preset: "balanced"
  n_trials: 200
```

### Advanced Configuration

For full control, use the explicit configuration format:

```yaml
optimization:
  approach: "single_objective"  # or "multi_objective"
  objective:  # For single objective
    metric: "output_tokens_per_second"
    direction: "maximize"
    percentile: "median"
  sampler: "tpe"
  n_trials: 100
```

## Available Metrics

### Primary Performance Metrics

| Metric | Description | Typical Goal | Units |
|--------|-------------|--------------|-------|
| `output_tokens_per_second` | Token generation throughput | Maximize | tokens/sec |
| `request_latency` | End-to-end request latency | Minimize | milliseconds |
| `time_to_first_token_ms` | Time until first token (TTFT) | Minimize | milliseconds |
| `inter_token_latency_ms` | Latency between tokens (ITL) | Minimize | milliseconds |
| `requests_per_second` | Request throughput | Maximize | requests/sec |

### Percentile Options

| Percentile | Description | When to Use |
|------------|-------------|-------------|
| `"median"` or `"p50"` | 50th percentile | Most common, stable optimization |
| `"p95"` | 95th percentile | SLA optimization, tail latency |
| `"p90"` | 90th percentile | Good balance between median and extreme cases |
| `"p99"` | 99th percentile | Extreme tail latency optimization |

## Optimization Approaches

### 1. Single Objective Optimization

Optimize for one metric only. Best when you have a clear primary goal.

#### Example: Maximize Throughput
```yaml
optimization:
  approach: "single_objective"
  objective:
    metric: "output_tokens_per_second"
    direction: "maximize"
    percentile: "median"
  sampler: "tpe"
  n_trials: 100
```

#### Example: Minimize P95 Latency
```yaml
optimization:
  approach: "single_objective"
  objective:
    metric: "request_latency"
    direction: "minimize"
    percentile: "p95"  # Optimize for 95th percentile SLA
  sampler: "tpe"
  n_trials: 100
```

#### Example: Minimize Time-To-First-Token
```yaml
optimization:
  approach: "single_objective"
  objective:
    metric: "time_to_first_token_ms"
    direction: "minimize"
    percentile: "p95"
  sampler: "tpe"
  n_trials: 100
```

### 2. Multi-Objective Optimization

Find optimal trade-offs between multiple metrics. Returns Pareto-optimal solutions.

#### Example: Throughput vs Latency (Most Common)
```yaml
optimization:
  approach: "multi_objective"
  objectives:
    - metric: "output_tokens_per_second"
      direction: "maximize"
      percentile: "median"
    - metric: "request_latency"
      direction: "minimize"
      percentile: "median"
  sampler: "nsga2"  # Recommended for multi-objective
  n_trials: 200
```

#### Example: Throughput vs TTFT
```yaml
optimization:
  approach: "multi_objective"
  objectives:
    - metric: "output_tokens_per_second"
      direction: "maximize"
      percentile: "median"
    - metric: "time_to_first_token_ms"
      direction: "minimize"
      percentile: "p95"
  sampler: "nsga2"
  n_trials: 200
```

#### Example: TTFT vs End-to-End Latency
```yaml
optimization:
  approach: "multi_objective"
  objectives:
    - metric: "time_to_first_token_ms"
      direction: "minimize"
      percentile: "p95"
    - metric: "request_latency"
      direction: "minimize"
      percentile: "p95"
  sampler: "nsga2"
  n_trials: 200
```

## Sampler Selection

| Sampler | Best For | Description |
|---------|----------|-------------|
| `"tpe"` | Single objective | Tree-structured Parzen Estimator, good default |
| `"nsga2"` | Multi-objective | Non-dominated Sorting Genetic Algorithm II |
| `"botorch"` | Single objective (advanced) | Bayesian optimization, slower but powerful |
| `"random"` | Testing/baseline | Random sampling, good for quick tests |
| `"grid"` | Exhaustive search | Grid search over all parameter combinations |

## Use Case Examples

### High-Performance Serving
```yaml
# Focus on maximum throughput
optimization:
  preset: "high_throughput"
  # OR explicit:
  # approach: "single_objective"
  # objective:
  #   metric: "output_tokens_per_second"
  #   direction: "maximize"
  #   percentile: "median"
```

### Interactive Applications
```yaml
# Minimize time to first token for responsiveness
optimization:
  approach: "single_objective"
  objective:
    metric: "time_to_first_token_ms"
    direction: "minimize"
    percentile: "p95"
  sampler: "tpe"
  n_trials: 100
```

### Production SLA Optimization
```yaml
# Minimize P95 latency for SLA compliance
optimization:
  approach: "single_objective"
  objective:
    metric: "request_latency"
    direction: "minimize"
    percentile: "p95"
  sampler: "tpe"
  n_trials: 150
```

### Balanced Production Setup
```yaml
# Find best throughput vs latency trade-offs
optimization:
  preset: "balanced"
  # OR explicit:
  # approach: "multi_objective"
  # objectives:
  #   - metric: "output_tokens_per_second"
  #     direction: "maximize"
  #     percentile: "median"
  #   - metric: "request_latency"
  #     direction: "minimize"
  #     percentile: "median"
```

### Streaming Applications
```yaml
# Optimize inter-token latency for smooth streaming
optimization:
  approach: "single_objective"
  objective:
    metric: "inter_token_latency_ms"
    direction: "minimize"
    percentile: "p95"
  sampler: "tpe"
  n_trials: 100
```

## Migration from Old Format

### Old Format (Deprecated but still supported)
```yaml
optimization:
  objective: "maximize"  # Vague, only optimizes throughput
  sampler: "tpe"
  n_trials: 100
```

### New Format (Recommended)
```yaml
optimization:
  approach: "single_objective"
  objective:
    metric: "output_tokens_per_second"  # Clear what's being optimized
    direction: "maximize"
    percentile: "median"
  sampler: "tpe"
  n_trials: 100

# OR use preset for simplicity
optimization:
  preset: "high_throughput"
  n_trials: 100
```

## Advanced Tips

### 1. Choosing Percentiles
- **Median (p50)**: Most stable, good for general optimization
- **P95**: Good for SLA requirements and tail latency
- **P99**: Use sparingly, can be noisy and lead to overfitting

### 2. Trial Count Guidelines
- **Single objective**: 50-150 trials usually sufficient
- **Multi-objective**: 150-300 trials for good Pareto frontier
- **Quick testing**: 10-20 trials with random sampler

### 3. Multi-Objective Interpretation
Multi-objective optimization returns multiple solutions on the Pareto frontier. Each solution represents a different trade-off. Review the results and choose the solution that best fits your requirements.

### 4. Parameter Space Considerations
More complex parameter spaces need more trials. If you have many parameters enabled, increase `n_trials` accordingly.

## Troubleshooting

### Issue: Optimization seems random
- **Solution**: Increase `n_trials`, ensure parameters have reasonable ranges
- **Check**: Parameter space isn't too large relative to trial count

### Issue: No improvement over baseline
- **Solution**: Check if parameters are actually affecting the chosen metric
- **Try**: Different metrics or multi-objective approach

### Issue: Multi-objective results unclear
- **Solution**: Use visualization tools to explore Pareto frontier
- **Consider**: Single objective if one metric is clearly most important

## Examples in This Repository

- `study_config_local_exec.yaml`: High-throughput optimization
- `study_config_no_postgres.yaml`: Low-latency optimization  
- `study_config_minimal.yaml`: Balanced multi-objective
- `study_config.yaml`: Advanced throughput vs TTFT optimization
- `study_config_optimization_examples.yaml`: Comprehensive examples
