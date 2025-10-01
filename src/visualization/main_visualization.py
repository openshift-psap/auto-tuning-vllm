import json
import os

import optuna
import plotly.graph_objects as go

# Import all optuna visualization modules
from optuna.visualization import plot_optimization_history
from plotly.subplots import make_subplots


def load_baseline_metrics(study_dir, study_id):
    """Load baseline metrics from multiple concurrency JSON files."""
    baseline_files = []
    baselines = {}

    # Look for baseline files with concurrency patterns
    import glob

    baseline_pattern = os.path.join(
        study_dir, f"benchmarks_{study_id}.baseline.concurrency_*.json"
    )
    baseline_files = glob.glob(baseline_pattern)

    # If no concurrency-specific baselines found, look for the old format
    if not baseline_files:
        old_baseline_file = os.path.join(
            study_dir, f"benchmarks_{study_id}.baseline.json"
        )
        if os.path.exists(old_baseline_file):
            baseline_files = [old_baseline_file]

    if not baseline_files:
        print(f"Warning: No baseline files found in {study_dir}")
        return {}

    for baseline_file in baseline_files:
        # Extract concurrency from filename
        if "concurrency_" in baseline_file:
            concurrency = int(baseline_file.split("concurrency_")[1].split(".json")[0])
        else:
            concurrency = 50  # Default for old format

        try:
            with open(baseline_file, "r") as f:
                data = json.load(f)

            # Extract metrics from the benchmark data structure
            stats = data["benchmarks"][0]
            metrics = stats["metrics"]

            baseline_metrics = {}
            baseline_metrics["output_tokens_per_second"] = metrics[
                "output_tokens_per_second"
            ]["successful"]["median"]
            baseline_metrics["request_latency"] = metrics["request_latency"][
                "successful"
            ]["median"]
            baseline_metrics["time_to_first_token_ms"] = metrics[
                "time_to_first_token_ms"
            ]["successful"]["median"]
            baseline_metrics["tokens_per_second"] = metrics["tokens_per_second"][
                "successful"
            ]["median"]
            baseline_metrics["concurrency"] = concurrency

            baselines[concurrency] = baseline_metrics

        except Exception as e:
            print(f"Error loading baseline metrics from {baseline_file}: {e}")
            continue

    return baselines


def create_single_objective_visualization(study, baseline_metrics, study_dir, study_id):
    """Create visualization for single-objective optimization."""
    print("Creating single-objective optimization visualization...")

    # Create visualization directory
    viz_dir = os.path.join(study_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Create optimization history plot
    fig = plot_optimization_history(study)

    # Add baseline as horizontal line if available
    if baseline_metrics:
        # Show the first baseline by default
        first_concurrency = min(baseline_metrics.keys())
        baseline_value = baseline_metrics[first_concurrency].get(
            "output_tokens_per_second", 0
        )
        fig.add_hline(
            y=baseline_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Baseline (C={first_concurrency}): {baseline_value:.2f} tokens/s",
            annotation_position="top right",
        )

    fig.update_layout(
        title=f"Optimization History - Study {study_id}",
        xaxis_title="Trial",
        yaxis_title="Output Tokens per Second",
    )

    # Save HTML
    html_path = os.path.join(viz_dir, f"optimization_history_{study_id}.html")

    fig.write_html(html_path)
    saved_files = [html_path]

    print("Single-objective visualization saved:")
    print(f"  HTML: {html_path}")

    return saved_files


def create_multi_objective_visualizations(study, baseline_metrics, study_dir, study_id):
    """Create visualizations for multi-objective optimization."""
    print("Creating multi-objective optimization visualizations...")

    # Create visualization directory
    viz_dir = os.path.join(study_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    saved_files = []

    # Get completed trials
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        print("No completed trials found for visualization.")
        return []

    # Extract objective values and trial numbers
    trial_numbers = [trial.number for trial in completed_trials]
    throughputs = [
        trial.values[0] for trial in completed_trials
    ]  # First objective: throughput
    latencies = [
        trial.values[1] for trial in completed_trials
    ]  # Second objective: latency

    # Extract additional metrics from user_attrs for comprehensive hover info
    trial_metrics = []
    for trial in completed_trials:
        metrics = {}
        if hasattr(trial, "user_attrs") and trial.user_attrs:
            # Extract common guidellm metrics
            metrics["output_tokens_per_second"] = trial.user_attrs.get(
                "output_tokens_per_second", trial.values[0]
            )
            metrics["request_latency"] = trial.user_attrs.get(
                "request_latency", trial.values[1] if len(trial.values) > 1 else 0
            )
            metrics["time_to_first_token_ms"] = trial.user_attrs.get(
                "time_to_first_token_ms", "N/A"
            )
            metrics["tokens_per_second"] = trial.user_attrs.get(
                "tokens_per_second", "N/A"
            )
            metrics["inter_token_latency_ms"] = trial.user_attrs.get(
                "inter_token_latency_ms", "N/A"
            )
            metrics["output_sequence_length"] = trial.user_attrs.get(
                "output_sequence_length", "N/A"
            )
            metrics["input_sequence_length"] = trial.user_attrs.get(
                "input_sequence_length", "N/A"
            )
            # Extract concurrency from trial parameters
            metrics["concurrency"] = trial.params.get("guidellm_concurrency", "N/A")
        else:
            # Fallback to basic metrics
            metrics["output_tokens_per_second"] = trial.values[0]
            metrics["request_latency"] = trial.values[1] if len(trial.values) > 1 else 0
            metrics["time_to_first_token_ms"] = "N/A"
            metrics["tokens_per_second"] = "N/A"
            metrics["inter_token_latency_ms"] = "N/A"
            metrics["output_sequence_length"] = "N/A"
            metrics["input_sequence_length"] = "N/A"
            metrics["concurrency"] = trial.params.get("guidellm_concurrency", "N/A")
        trial_metrics.append(metrics)

    # 1. Throughput optimization history
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=trial_numbers,
            y=throughputs,
            mode="lines+markers",
            name="Throughput",
            line=dict(color="blue"),
        )
    )

    # Add all baseline throughputs as separate traces
    if baseline_metrics:
        for i, (concurrency, metrics) in enumerate(baseline_metrics.items()):
            baseline_throughput = metrics.get("output_tokens_per_second", 0)
            fig1.add_trace(
                go.Scatter(
                    x=trial_numbers,
                    y=[baseline_throughput] * len(trial_numbers),
                    mode="lines",
                    name=f"Baseline C={concurrency}",
                    line=dict(color="red", dash="dash"),
                    visible=True
                    if i == 0
                    else False,  # Show only the first one initially
                    hovertemplate=f"Baseline (C={concurrency}): {baseline_throughput:.2f} tokens/s<extra></extra>",
                )
            )

    fig1.update_layout(
        title=f"Throughput Optimization History - Study {study_id}",
        xaxis_title="Trial",
        yaxis_title="Output Tokens per Second",
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                buttons=list(
                    [
                        dict(
                            args=[
                                {
                                    "visible": [True]
                                    + [
                                        concurrency == c
                                        for c in sorted(baseline_metrics.keys())
                                    ]
                                }
                            ],
                            label=f"Baseline C={concurrency}",
                            method="restyle",
                        )
                        for concurrency in sorted(baseline_metrics.keys())
                    ]
                )
                if baseline_metrics
                else [],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.99,
                xanchor="right",
                y=1.25,
                yanchor="top",
            ),
        ]
        if baseline_metrics
        else [],
    )

    html_path1 = os.path.join(viz_dir, f"throughput_history_{study_id}.html")
    fig1.write_html(html_path1)
    saved_files.append(html_path1)

    # 2. Latency optimization history
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=trial_numbers,
            y=latencies,
            mode="lines+markers",
            name="Latency",
            line=dict(color="orange"),
        )
    )

    # Add all baseline latencies as separate traces
    if baseline_metrics:
        for i, (concurrency, metrics) in enumerate(baseline_metrics.items()):
            baseline_latency = metrics.get("request_latency", 0)
            fig2.add_trace(
                go.Scatter(
                    x=trial_numbers,
                    y=[baseline_latency] * len(trial_numbers),
                    mode="lines",
                    name=f"Baseline C={concurrency}",
                    line=dict(color="red", dash="dash"),
                    visible=True
                    if i == 0
                    else False,  # Show only the first one initially
                    hovertemplate=f"Baseline (C={concurrency}): {baseline_latency:.2f} ms<extra></extra>",
                )
            )

    fig2.update_layout(
        title=f"Latency Optimization History - Study {study_id}",
        xaxis_title="Trial",
        yaxis_title="Request Latency (ms)",
        yaxis=dict(range=[0, None]),  # Make latency start from 0
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                buttons=list(
                    [
                        dict(
                            args=[
                                {
                                    "visible": [True]
                                    + [
                                        concurrency == c
                                        for c in sorted(baseline_metrics.keys())
                                    ]
                                }
                            ],
                            label=f"Baseline C={concurrency}",
                            method="restyle",
                        )
                        for concurrency in sorted(baseline_metrics.keys())
                    ]
                )
                if baseline_metrics
                else [],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.99,
                xanchor="right",
                y=1.25,
                yanchor="top",
            ),
        ]
        if baseline_metrics
        else [],
    )

    html_path2 = os.path.join(viz_dir, f"latency_history_{study_id}.html")
    fig2.write_html(html_path2)
    saved_files.append(html_path2)

    # 3. Combined 2D line graph with both metrics
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])

    # Create comprehensive hover text for each trial
    hover_texts = []
    for i, (trial_num, metrics) in enumerate(zip(trial_numbers, trial_metrics)):
        hover_text = f"<b>Trial {trial_num}</b><br>"
        hover_text += f"Output Tokens/s: {metrics['output_tokens_per_second']:.2f}<br>"
        hover_text += f"Request Latency: {metrics['request_latency']:.2f} ms<br>"
        hover_text += f"Concurrency: {metrics['concurrency']}<br>"
        if metrics["time_to_first_token_ms"] != "N/A":
            hover_text += f"TTFT: {metrics['time_to_first_token_ms']:.2f} ms<br>"
        if metrics["tokens_per_second"] != "N/A":
            hover_text += f"Total Tokens/s: {metrics['tokens_per_second']:.2f}<br>"
        if metrics["inter_token_latency_ms"] != "N/A":
            hover_text += (
                f"Inter-token Latency: {metrics['inter_token_latency_ms']:.2f} ms<br>"
            )
        if metrics["output_sequence_length"] != "N/A":
            hover_text += f"Output Length: {metrics['output_sequence_length']}<br>"
        if metrics["input_sequence_length"] != "N/A":
            hover_text += f"Input Length: {metrics['input_sequence_length']}"
        hover_texts.append(hover_text)

    # Plot throughput line
    fig3.add_trace(
        go.Scatter(
            x=trial_numbers,
            y=throughputs,
            mode="lines+markers",
            name="Throughput",
            line=dict(color="red", width=2),
            marker=dict(size=6),
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts,
        ),
        secondary_y=False,
    )

    # Plot latency line on secondary y-axis
    fig3.add_trace(
        go.Scatter(
            x=trial_numbers,
            y=latencies,
            mode="lines+markers",
            name="Latency",
            line=dict(color="orange", width=2),
            marker=dict(size=6),
            yaxis="y2",
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts,
        ),
        secondary_y=True,
    )

    # Add baseline horizontal lines for each concurrency level as separate traces
    if baseline_metrics:
        for i, (concurrency, metrics) in enumerate(baseline_metrics.items()):
            baseline_throughput = metrics.get("output_tokens_per_second", 0)
            baseline_latency = metrics.get("request_latency", 0)

            # Add baseline throughput line
            fig3.add_trace(
                go.Scatter(
                    x=trial_numbers,
                    y=[baseline_throughput] * len(trial_numbers),
                    mode="lines",
                    name=f"Baseline Throughput C={concurrency}",
                    line=dict(color="red", dash="dash"),
                    visible=True
                    if i == 0
                    else False,  # Show only the first one initially
                    hovertemplate=f"Baseline Throughput (C={concurrency}): {baseline_throughput:.2f} tokens/s<extra></extra>",
                    showlegend=False,
                ),
                secondary_y=False,
            )

            # Add baseline latency line
            fig3.add_trace(
                go.Scatter(
                    x=trial_numbers,
                    y=[baseline_latency] * len(trial_numbers),
                    mode="lines",
                    name=f"Baseline Latency C={concurrency}",
                    line=dict(color="orange", dash="dash"),
                    yaxis="y2",
                    visible=True
                    if i == 0
                    else False,  # Show only the first one initially
                    hovertemplate=f"Baseline Latency (C={concurrency}): {baseline_latency:.2f} ms<extra></extra>",
                    showlegend=False,
                ),
                secondary_y=True,
            )

    # Set x-axis title
    fig3.update_xaxes(title_text="Trial")

    # Set y-axes titles and ranges
    fig3.update_yaxes(
        title_text="Output Tokens per Second", range=[0, None], secondary_y=False
    )
    fig3.update_yaxes(
        title_text="Request Latency (ms)", range=[0, None], secondary_y=True
    )  # Make latency start from 0

    # Add baseline selector dropdown
    baseline_buttons = []
    if baseline_metrics:
        for concurrency in sorted(baseline_metrics.keys()):
            # Create visibility list for this concurrency
            visibility = [True, True]  # Always show throughput and latency main traces

            # Add baseline visibility (2 traces per baseline: throughput and latency)
            for c in sorted(baseline_metrics.keys()):
                visibility.extend(
                    [concurrency == c, concurrency == c]
                )  # Two lines per baseline

            baseline_buttons.append(
                dict(
                    args=[{"visible": visibility}],
                    label=f"Baseline C={concurrency}",
                    method="restyle",
                )
            )

    fig3.update_layout(
        title=f"Multi-Objective Optimization: Combined Metrics - Study {study_id}",
        hovermode="x unified",
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                buttons=baseline_buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.99,
                xanchor="right",
                y=1.25,
                yanchor="top",
            ),
        ]
        if baseline_metrics
        else [],
    )

    html_path3 = os.path.join(viz_dir, f"multi_objective_visualization_{study_id}.html")
    fig3.write_html(html_path3)
    saved_files.append(html_path3)

    print("Multi-objective visualizations saved:")
    for file_path in saved_files:
        print(f"  {file_path}")

    return saved_files


def main():
    """Main function to run the visualization program."""
    print("=== Optuna Study Visualization Tool ===")

    # Get study number from user
    while True:
        try:
            study_id = input("Enter the study number: ").strip()
            if not study_id:
                print("Please enter a valid study number.")
                continue
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            return

    # Construct paths
    current_dir = os.getcwd()

    # Find the project root by looking for src directory
    project_root = current_dir
    while project_root != os.path.dirname(project_root):  # Stop at filesystem root
        if os.path.exists(os.path.join(project_root, "src")):
            break
        project_root = os.path.dirname(project_root)

    # If we're already in src/visualization, go up two levels
    if current_dir.endswith("src/visualization"):
        src_dir = os.path.dirname(current_dir)
    # If we're in src directory, use it directly
    elif current_dir.endswith("src"):
        src_dir = current_dir
    # Otherwise, assume we're in project root and look for src
    else:
        src_dir = os.path.join(project_root, "src")

    study_dir = os.path.join(src_dir, "studies", f"study_{study_id}")
    db_path = os.path.join(study_dir, "optuna.db")

    # Check if study exists
    if not os.path.exists(db_path):
        print(f"Error: Study database not found at {db_path}")
        print("Please make sure the study number is correct.")
        return

    print(f"Loading study from: {db_path}")

    try:
        # Load the study
        storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}")

        # Try to find the study name - it might be different formats
        study_names_to_try = [f"study_{study_id}", f"study{study_id}", study_id]
        study = None

        for study_name in study_names_to_try:
            try:
                study = optuna.load_study(storage=storage, study_name=study_name)
                print(f"Found study with name: {study_name}")
                break
            except KeyError:
                continue

        if study is None:
            # List all available studies
            try:
                all_studies = optuna.study.get_all_study_summaries(storage)
                print("Available studies:")
                for summary in all_studies:
                    print(f"  - {summary.study_name}")
                if all_studies:
                    # Use the first study if only one exists
                    if len(all_studies) == 1:
                        study = optuna.load_study(
                            storage=storage, study_name=all_studies[0].study_name
                        )
                        print(
                            f"Using the only available study: {all_studies[0].study_name}"
                        )
                    else:
                        print("Multiple studies found. Please check the study name.")
                        return
                else:
                    print("No studies found in the database.")
                    return
            except Exception as e:
                print(f"Could not load study or list studies: {e}")
                return

        print("Study loaded successfully!")
        print(f"Study name: {study.study_name}")

        # Handle both single and multi-objective studies
        if len(study.directions) == 1:
            print(f"Direction: {study.direction}")
        else:
            print(f"Directions: {study.directions}")

        print(f"Total trials: {len(study.trials)}")

        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        print(f"Completed trials: {len(completed_trials)}")

        if not completed_trials:
            print("No completed trials found. Cannot create visualizations.")
            return

        # Load baseline metrics
        baseline_metrics = load_baseline_metrics(study_dir, study_id)
        if baseline_metrics:
            print("Baseline metrics loaded successfully!")
            for concurrency, metrics in baseline_metrics.items():
                print(
                    f"Baseline C={concurrency}: {metrics['output_tokens_per_second']:.2f} tokens/s, {metrics['request_latency']:.2f} ms"
                )
        else:
            print("Warning: Could not load baseline metrics.")

        # Determine if it's multi-objective
        is_multi_objective = len(study.directions) > 1

        print(
            f"\nOptimization type: {'Multi-objective' if is_multi_objective else 'Single-objective'}"
        )

        # Create appropriate visualizations
        if is_multi_objective:
            saved_files = create_multi_objective_visualizations(
                study, baseline_metrics, study_dir, study_id
            )
        else:
            saved_files = create_single_objective_visualization(
                study, baseline_metrics, study_dir, study_id
            )

        print(f"Visualization complete! {len(saved_files)} files saved.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
