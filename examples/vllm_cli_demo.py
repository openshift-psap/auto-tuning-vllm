#!/usr/bin/env python3
"""
Demo script showing how to use the vLLM CLI parser utility.

This script demonstrates:
1. Parsing vLLM CLI arguments
2. Generating parameter schemas
3. Comparing with existing schemas
4. Creating configuration files
"""

import json
import yaml
from pathlib import Path

from auto_tune_vllm.utils import VLLMCLIParser, ArgumentType


def main():
    """Demo the vLLM CLI parser functionality."""
    print("🚀 vLLM CLI Parser Demo")
    print("=" * 50)
    
    # Initialize parser with venv path
    parser = VLLMCLIParser(venv_path=".venv")
    
    print("📋 Parsing vLLM CLI arguments...")
    try:
        arguments = parser.parse()
        print(f"✅ Successfully parsed {len(arguments)} arguments")
    except Exception as e:
        print(f"❌ Failed to parse: {e}")
        return
    
    # Show section breakdown
    print("\n📊 Arguments by section:")
    total_args = 0
    for section in sorted(parser.get_all_sections()):
        args_in_section = parser.get_arguments_by_section(section)
        print(f"   {section:20} {len(args_in_section):3d} arguments")
        total_args += len(args_in_section)
    print(f"   {'TOTAL':20} {total_args:3d} arguments")
    
    # Show examples of different argument types
    print("\n🔍 Examples by argument type:")
    type_examples = {}
    for arg in arguments.values():
        if arg.arg_type not in type_examples:
            type_examples[arg.arg_type] = []
        if len(type_examples[arg.arg_type]) < 3:  # Keep only 3 examples per type
            type_examples[arg.arg_type].append(arg)
    
    for arg_type, examples in type_examples.items():
        print(f"\n   {arg_type.value.upper()}:")
        for arg in examples:
            print(f"     {arg.long_name:30} default={arg.default_value}")
            if arg.choices:
                print(f"     {' ':30} choices={arg.choices[:5]}")  # Show first 5 choices
    
    # Focus on optimization-relevant sections
    optimization_sections = ['cacheconfig', 'schedulerconfig', 'modelconfig', 'parallelconfig']
    
    print("\n⚙️  Optimization-relevant arguments:")
    optimization_args = {}
    for section in optimization_sections:
        section_args = parser.get_arguments_by_section(section)
        optimization_args[section] = section_args
        print(f"   {section:15} {len(section_args):3d} arguments")
    
    # Generate parameter schema for optimization sections
    print("\n🔧 Generating parameter schema...")
    schema = parser.generate_parameter_schema(optimization_sections)
    
    # Save schema to file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    schema_file = output_dir / "vllm_generated_schema.yaml"
    with open(schema_file, 'w') as f:
        yaml.dump(schema, f, default_flow_style=False, sort_keys=True)
    print(f"💾 Saved parameter schema to: {schema_file}")
    
    # Save full arguments data as JSON
    json_file = output_dir / "vllm_cli_arguments.json"
    with open(json_file, 'w') as f:
        json.dump(parser.to_dict(), f, indent=2, default=str)
    print(f"💾 Saved full argument data to: {json_file}")
    
    # Export defaults YAML for configuration loading
    print("\n🎯 Exporting vLLM defaults...")
    defaults_file = output_dir / "vllm_defaults.yaml"
    defaults_data = parser.export_defaults_yaml(str(defaults_file), optimization_sections)
    print(f"💾 Saved vLLM defaults to: {defaults_file}")
    
    # Show some examples of extracted defaults
    total_defaults = sum(len(section_defaults) for section_defaults in defaults_data.get('defaults', {}).values())
    print(f"📊 Extracted {total_defaults} default values across {len(defaults_data.get('defaults', {}))} sections")
    
    # Show sample defaults
    print("\n🔍 Sample extracted defaults:")
    for section, section_defaults in defaults_data.get('defaults', {}).items():
        print(f"   {section}:")
        sample_count = 0
        for param, default in section_defaults.items():
            if sample_count < 3:  # Show first 3 per section
                print(f"     {param:25} = {default}")
                sample_count += 1
        if len(section_defaults) > 3:
            print(f"     {'... and ' + str(len(section_defaults) - 3) + ' more':25}")
    
    # Compare with existing schema
    existing_schema_path = Path("auto_tune_vllm/schemas/parameter_schema.yaml")
    if existing_schema_path.exists():
        print("\n🔄 Comparing with existing schema...")
        with open(existing_schema_path) as f:
            existing_schema = yaml.safe_load(f)
        
        existing_params = set(existing_schema.get('parameters', {}).keys())
        generated_params = set(schema['parameters'].keys())
        
        print(f"   Existing schema:  {len(existing_params):3d} parameters")
        print(f"   Generated schema: {len(generated_params):3d} parameters")
        
        # Find overlaps and differences
        common = existing_params & generated_params
        only_existing = existing_params - generated_params
        only_generated = generated_params - existing_params
        
        print(f"   Common:           {len(common):3d} parameters")
        print(f"   Only in existing: {len(only_existing):3d} parameters")
        print(f"   Only in generated:{len(only_generated):3d} parameters")
        
        if only_existing:
            print("\n   📋 Parameters only in existing schema:")
            for param in sorted(only_existing)[:10]:  # Show first 10
                print(f"      {param}")
            if len(only_existing) > 10:
                print(f"      ... and {len(only_existing) - 10} more")
        
        if only_generated:
            print("\n   🆕 New parameters from vLLM CLI:")
            for param in sorted(only_generated)[:10]:  # Show first 10
                print(f"      {param}")
            if len(only_generated) > 10:
                print(f"      ... and {len(only_generated) - 10} more")
    
    # Generate example configuration
    print("\n📝 Generating example configuration...")
    example_config = generate_example_config(parser, optimization_sections)
    
    config_file = output_dir / "vllm_example_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False)
    print(f"💾 Saved example config to: {config_file}")
    
    print("\n✨ Demo completed! Check the 'output' directory for generated files.")


def generate_example_config(parser: VLLMCLIParser, sections: list) -> dict:
    """Generate an example configuration file based on parsed arguments."""
    config = {
        "# Example vLLM configuration generated from CLI parsing": None,
        "study": {
            "name": "vllm_cli_optimization",
            "database_url": "postgresql://user:pass@localhost/optuna"
        },
        "optimization": {
            "approach": "single_objective",
            "objectives": [
                {"metric": "requests_per_second", "direction": "maximize"}
            ],
            "sampler": "tpe",
            "n_trials": 100
        },
        "benchmark": {
            "benchmark_type": "guidellm",
            "model": "microsoft/DialoGPT-medium",
            "max_seconds": 300,
            "use_synthetic_data": True,
            "prompt_tokens": 1000,
            "output_tokens": 500
        },
        "logging": {
            "file_path": "/tmp/auto-tune-vllm-logs",
            "log_level": "INFO"
        },
        "parameters": {}
    }
    
    # Add some interesting parameters from each section
    interesting_params = {
        'cacheconfig': ['gpu_memory_utilization', 'block_size', 'kv_cache_dtype'],
        'schedulerconfig': ['max_num_batched_tokens', 'max_num_seqs', 'cuda_graph_sizes'],
        'modelconfig': ['max_model_len', 'dtype', 'enforce_eager'],
        'parallelconfig': ['tensor_parallel_size', 'pipeline_parallel_size']
    }
    
    for section in sections:
        if section not in interesting_params:
            continue
            
        section_args = {arg.long_name[2:].replace('-', '_'): arg 
                       for arg in parser.get_arguments_by_section(section)}
        
        for param_name in interesting_params[section]:
            if param_name in section_args:
                arg = section_args[param_name]
                param_config = {"enabled": True}
                
                # Add some example parameter bounds or choices
                if arg.arg_type == ArgumentType.INTEGER and arg.default_value is not None:
                    param_config.update({
                        "min": max(1, arg.default_value // 2),
                        "max": arg.default_value * 2,
                        "step": max(1, arg.default_value // 10)
                    })
                elif arg.arg_type == ArgumentType.FLOAT and arg.default_value is not None:
                    param_config.update({
                        "min": max(0.1, arg.default_value - 0.2),
                        "max": min(1.0, arg.default_value + 0.2),
                        "step": 0.05
                    })
                elif arg.arg_type == ArgumentType.CHOICE and arg.choices:
                    param_config["options"] = arg.choices[:3]  # Use first 3 choices
                elif arg.arg_type == ArgumentType.BOOLEAN:
                    # For boolean, just enable it
                    pass
                
                config["parameters"][param_name] = param_config
    
    # Clean up the None values
    if None in config:
        del config[None]
    
    return config


if __name__ == "__main__":
    main()
