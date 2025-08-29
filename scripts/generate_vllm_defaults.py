#!/usr/bin/env python3
"""
Script to generate vLLM defaults.yaml from CLI parsing.

This script extracts default values from 'vllm serve --help' and saves them
to a YAML file that can be used by the configuration system instead of 
hardcoding values.

Usage:
    # Activate virtual environment and run:
    source .venv/bin/activate
    python scripts/generate_vllm_defaults.py [--output path] [--sections section1,section2,...]
    
    # Or install the package in development mode:
    pip install -e .
    python scripts/generate_vllm_defaults.py [--help]
"""

import argparse
import sys
from pathlib import Path

try:
    from auto_tune_vllm.utils import VLLMCLIParser
except ImportError as e:
    print(f"Error importing VLLMCLIParser: {e}")
    print("Make sure to:")
    print("1. Activate the virtual environment: source .venv/bin/activate") 
    print("2. Install the package: pip install -e .")
    print("3. Then run: python scripts/generate_vllm_defaults.py")
    sys.exit(1)


def main():
    """Generate vLLM defaults YAML file."""
    parser = argparse.ArgumentParser(
        description="Generate vLLM defaults.yaml from CLI parsing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for defaults YAML file (default: auto_tune_vllm/schemas/vllm_defaults/v{version}.yaml)"
    )
    
    parser.add_argument(
        "--versioned", 
        action="store_true",
        default=True,
        help="Generate versioned defaults file (default: True)"
    )
    
    parser.add_argument(
        "--no-versioned",
        action="store_false",
        dest="versioned",
        help="Generate single defaults file instead of versioned"
    )
    
    parser.add_argument(
        "--sections", "-s",
        default="cacheconfig,schedulerconfig,modelconfig,parallelconfig",
        help="Comma-separated list of CLI sections to include (default: optimization-relevant sections)"
    )
    
    parser.add_argument(
        "--venv", "-v",
        default=".venv",
        help="Path to virtual environment (default: .venv)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Parse sections
    if args.sections.lower() == "all":
        sections = None  # Include all sections
    else:
        sections = [s.strip() for s in args.sections.split(",")]
    
    # Initialize parser and get version
    cli_parser = VLLMCLIParser(venv_path=args.venv)
    vllm_version = cli_parser.get_vllm_version()
    
    print("🔧 Generating vLLM defaults from CLI...")
    print(f"   Virtual environment: {args.venv}")
    print(f"   vLLM version: {vllm_version}")
    print(f"   Versioned: {args.versioned}")
    print(f"   Sections: {sections or 'all'}")
    
    try:
        # Parse CLI arguments
        if args.verbose:
            print("\n📋 Parsing vLLM CLI arguments...")
        arguments = cli_parser.parse()
        
        if args.verbose:
            print(f"✅ Parsed {len(arguments)} arguments")
            for section in cli_parser.get_all_sections():
                count = len(cli_parser.get_arguments_by_section(section))
                print(f"   {section}: {count} arguments")
        
        # Determine output path
        if args.versioned:
            if args.output:
                # Use provided path as base directory
                output_path = cli_parser.export_versioned_defaults(args.output, sections)
            else:
                # Use default versioned location
                output_path = cli_parser.export_versioned_defaults(
                    "auto_tune_vllm/schemas/vllm_defaults", sections
                )
            print("\n💾 Generating versioned defaults...")
        else:
            # Single file mode
            if args.output is None:
                args.output = "auto_tune_vllm/schemas/vllm_defaults.yaml"
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            defaults_data = cli_parser.export_defaults_yaml(str(output_path), sections)
            print("\n💾 Generating single defaults file...")
            
        # Show results
        if args.versioned:
            # Load the generated file to show stats
            with open(output_path) as f:
                import yaml
                defaults_data = yaml.safe_load(f)
        
        total_defaults = sum(len(section_defaults) for section_defaults in defaults_data.get('defaults', {}).values())
        print(f"✅ Successfully exported {total_defaults} default values")
        
        if args.verbose:
            print("\n📊 Defaults by section:")
            for section, section_defaults in defaults_data.get('defaults', {}).items():
                print(f"   {section}: {len(section_defaults)} defaults")
                
                # Show a few examples
                for i, (param, value) in enumerate(section_defaults.items()):
                    if i < 3:  # Show first 3
                        print(f"     {param} = {value}")
                if len(section_defaults) > 3:
                    print(f"     ... and {len(section_defaults) - 3} more")
        
        print(f"\n✨ Defaults file saved to: {output_path}")
        
        if args.versioned:
            print("   Also available as: auto_tune_vllm/schemas/vllm_defaults/latest.yaml")
            print(f"   Use with: StudyConfig.from_file(config_path, vllm_version='{vllm_version}')")
        else:
            print(f"   Use with: ConfigValidator(defaults_path='{output_path}')")
        
        # Show version management info if versioned
        if args.versioned:
            try:
                from auto_tune_vllm.utils import VLLMVersionManager
                manager = VLLMVersionManager()
                versions = manager.list_available_versions()
                if len(versions) > 1:
                    print(f"\n📚 Version history ({len(versions)} versions available):")
                    for i, version in enumerate(versions[:5]):  # Show first 5
                        marker = "→ " if i == 0 else "  "
                        print(f"   {marker}v{version.version} ({version.total_defaults} defaults)")
                    if len(versions) > 5:
                        print(f"   ... and {len(versions) - 5} more versions")
            except Exception as e:
                if args.verbose:
                    print(f"   Could not load version info: {e}")
        
    except Exception as e:
        print(f"❌ Error generating defaults: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
