"""
Utility to parse vLLM CLI arguments and defaults from 'vllm serve --help' output.

This module provides functionality to:
1. Execute 'vllm serve --help' and parse the output
2. Extract argument names, types, defaults, and descriptions
3. Organize arguments by their configuration groups
4. Generate structured data for use in configuration schemas
"""

import re
import subprocess
import sys
import json
import ast
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class ArgumentType(Enum):
    """Enumeration of argument types we can detect."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    CHOICE = "choice"
    JSON = "json"


@dataclass
class CLIArgument:
    """Represents a single CLI argument with all its properties."""
    long_name: str
    short_name: Optional[str] = None
    description: str = ""
    default_value: Any = None
    arg_type: ArgumentType = ArgumentType.STRING
    choices: Optional[List[str]] = None
    is_flag: bool = False
    section: str = "options"
    required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result['arg_type'] = result['arg_type'].value
        return result


class VLLMCLIParser:
    """Parser for vLLM CLI help output."""
    
    def __init__(self, venv_path: Optional[str] = None):
        """
        Initialize the parser.
        
        Args:
            venv_path: Path to virtual environment to use for vllm command
        """
        self.venv_path = venv_path
        self.arguments: Dict[str, CLIArgument] = {}
        self.sections: Dict[str, List[str]] = {}
        self._vllm_version: Optional[str] = None
    
    def get_vllm_version(self) -> str:
        """Get the vLLM version."""
        if self._vllm_version is not None:
            return self._vllm_version
        
        python_bin = f"{self.venv_path}/bin/python" if self.venv_path else sys.executable
        try:
            result = subprocess.run(
                [python_bin, "-m", "vllm", "-v"],
                capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as err:
            raise RuntimeError(
                f"Failed to get vLLM version (exit {err.returncode}): {err.stderr or err.stdout}"
            ) from err
        
        # Extract version from output (format: "0.10.1.1")
        version_line = result.stdout.strip().split('\n')[-1]
        self._vllm_version = version_line.strip()
        return self._vllm_version
    
    def get_help_output(self) -> str:
        """Execute 'vllm serve --help' and return the output."""
        python_bin = f"{self.venv_path}/bin/python" if self.venv_path else sys.executable
        try:
            result = subprocess.run(
                [python_bin, "-m", "vllm", "serve", "--help"],
                capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as err:
            raise RuntimeError(
                f"Failed to get vLLM help (exit {err.returncode}): {err.stderr or err.stdout}"
            ) from err
        
        return result.stdout
    
    def _extract_default_value(self, description: str) -> Tuple[Any, str]:
        """Extract default value from description text."""
        # Pattern to match (default: value) at end of description
        default_pattern = r'\(default:\s*([^)]+)\)\s*$'
        match = re.search(default_pattern, description)
        
        if not match:
            return None, description
        
        default_str = match.group(1).strip()
        cleaned_desc = re.sub(default_pattern, '', description).strip()
        
        # Parse different default value types
        if default_str.lower() in ['none', 'null']:
            return None, cleaned_desc
        elif default_str.lower() in ['true', 'false']:
            return default_str.lower() == 'true', cleaned_desc
        elif default_str.startswith('[') and default_str.endswith(']'):
            # List default
            try:
                return ast.literal_eval(default_str), cleaned_desc
            except Exception:
                return default_str, cleaned_desc
        elif default_str.startswith('{') and default_str.endswith('}'):
            # Dict/JSON default
            try:
                return ast.literal_eval(default_str), cleaned_desc
            except Exception:
                return default_str, cleaned_desc
        else:
            # Try to parse as number
            try:
                if '.' in default_str:
                    return float(default_str), cleaned_desc
                else:
                    return int(default_str), cleaned_desc
            except ValueError:
                return default_str, cleaned_desc
    
    def _detect_argument_type(self, arg_name: str, description: str, 
                             default_value: Any, choices: Optional[List[str]]) -> ArgumentType:
        """Detect the argument type based on various clues."""
        # Check for choices first
        if choices:
            return ArgumentType.CHOICE
        
        # Check for boolean flags
        if arg_name.startswith('--enable-') or arg_name.startswith('--disable-') or \
           arg_name.startswith('--no-') or 'enable' in description.lower() or \
           isinstance(default_value, bool):
            return ArgumentType.BOOLEAN
        
        # Check for JSON arguments
        if 'json' in description.lower() or 'config' in arg_name.lower():
            return ArgumentType.JSON
        
        # Check for lists
        if isinstance(default_value, list) or '[' in description or \
           'list' in description.lower() or arg_name.endswith('s'):
            return ArgumentType.LIST
        
        # Check for numbers
        if isinstance(default_value, int) or 'number' in description.lower() or \
           'count' in arg_name.lower() or 'size' in arg_name.lower():
            return ArgumentType.INTEGER
        
        if isinstance(default_value, float) or 'float' in description.lower() or \
           'ratio' in description.lower() or 'factor' in description.lower():
            return ArgumentType.FLOAT
        
        return ArgumentType.STRING
    
    def _extract_choices(self, description: str) -> Optional[List[str]]:
        """Extract choice options from description."""
        # Pattern for {choice1,choice2,choice3}
        choice_pattern = r'\{([^}]+)\}'
        match = re.search(choice_pattern, description)
        
        if match:
            choices_str = match.group(1)
            choices = [choice.strip() for choice in choices_str.split(',')]
            return choices
        
        return None
    
    def _parse_argument_line(self, line: str, section: str) -> Optional[CLIArgument]:
        """Parse a single argument line."""
        # Skip non-argument lines
        if not line.strip().startswith('--'):
            return None
        
        # Split argument definition from description
        parts = line.split(None, 1)  # Split on first whitespace
        if len(parts) < 2:
            return None
        
        arg_def = parts[0]
        description = parts[1] if len(parts) > 1 else ""
        
        # Parse argument name(s)
        arg_names = arg_def.split(',')
        long_name = None
        short_name = None
        
        for name in arg_names:
            name = name.strip()
            if name.startswith('--'):
                long_name = name
            elif name.startswith('-') and len(name) <= 4:  # Short options
                short_name = name
        
        if not long_name:
            return None
        
        # Extract default value and clean description
        default_value, clean_description = self._extract_default_value(description)
        
        # Extract choices
        choices = self._extract_choices(description)
        
        # Detect argument type
        arg_type = self._detect_argument_type(long_name, clean_description, default_value, choices)
        
        # Determine if it's a flag
        is_flag = arg_type == ArgumentType.BOOLEAN and default_value is not None
        
        return CLIArgument(
            long_name=long_name,
            short_name=short_name,
            description=clean_description,
            default_value=default_value,
            arg_type=arg_type,
            choices=choices,
            is_flag=is_flag,
            section=section
        )
    
    def parse_help_output(self, help_text: str) -> Dict[str, CLIArgument]:
        """Parse the complete help output into structured arguments."""
        lines = help_text.split('\n')
        current_section = "options"
        
        # Pattern to detect section headers
        section_pattern = r'^([A-Z][a-zA-Z]+):$'
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for section headers
            section_match = re.match(section_pattern, line)
            if section_match:
                current_section = section_match.group(1).lower()
                if current_section not in self.sections:
                    self.sections[current_section] = []
                i += 1
                continue
            
            # Check for argument lines
            if line.startswith('--'):
                # Collect multi-line argument description
                full_line = line
                j = i + 1
                while j < len(lines) and lines[j].startswith('                        '):
                    full_line += " " + lines[j].strip()
                    j += 1
                
                # Parse the argument
                arg = self._parse_argument_line(full_line, current_section)
                if arg:
                    self.arguments[arg.long_name] = arg
                    if current_section not in self.sections:
                        self.sections[current_section] = []
                    self.sections[current_section].append(arg.long_name)
                
                i = j
            else:
                i += 1
        
        return self.arguments
    
    def parse(self) -> Dict[str, CLIArgument]:
        """Parse vLLM CLI arguments by executing help command."""
        help_output = self.get_help_output()
        return self.parse_help_output(help_output)
    
    def get_arguments_by_section(self, section: str) -> List[CLIArgument]:
        """Get all arguments in a specific section."""
        if section not in self.sections:
            return []
        
        return [self.arguments[arg_name] for arg_name in self.sections[section] 
                if arg_name in self.arguments]
    
    def get_all_sections(self) -> List[str]:
        """Get list of all available sections."""
        return list(self.sections.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all parsed data to dictionary format."""
        return {
            'arguments': {name: arg.to_dict() for name, arg in self.arguments.items()},
            'sections': self.sections
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def generate_parameter_schema(self, sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a parameter schema similar to the existing parameter_schema.yaml.
        
        Args:
            sections: List of sections to include. If None, includes all.
        
        Returns:
            Dictionary in the format of parameter_schema.yaml
        """
        if sections is None:
            sections = self.get_all_sections()
        
        schema = {"parameters": {}}
        
        for section in sections:
            for arg in self.get_arguments_by_section(section):
                # Skip positional arguments and some special cases
                if not arg.long_name.startswith('--'):
                    continue
                
                # Convert --arg-name to arg_name
                param_name = arg.long_name[2:].replace('-', '_')
                
                param_def = {
                    "description": arg.description
                }
                
                if arg.arg_type == ArgumentType.CHOICE:
                    param_def["type"] = "list"
                    param_def["data_type"] = "str"
                    param_def["options"] = arg.choices
                elif arg.arg_type == ArgumentType.BOOLEAN:
                    param_def["type"] = "boolean"
                    param_def["data_type"] = "bool"
                elif arg.arg_type == ArgumentType.INTEGER:
                    param_def["type"] = "range"
                    param_def["data_type"] = "int"
                    # Try to infer reasonable min/max from description or defaults
                    if arg.default_value is not None:
                        param_def["min"] = max(0, arg.default_value - 100)
                        param_def["max"] = arg.default_value + 1000
                        param_def["step"] = 1
                elif arg.arg_type == ArgumentType.FLOAT:
                     param_def["type"] = "range"
                     param_def["data_type"] = "float"
                     if arg.default_value is not None:
                        dv = float(arg.default_value)
                        if 0.0 <= dv <= 1.0:
                            param_def["min"] = max(0.0, dv - 0.1)
                            param_def["max"] = min(1.0, dv + 0.1)
                        else:
                            span = max(1.0, abs(dv) * 0.25)
                            param_def["min"] = dv - span
                            param_def["max"] = dv + span
                        param_def["step"] = 0.01
                elif arg.arg_type == ArgumentType.LIST:
                    param_def["type"] = "list"
                    param_def["data_type"] = "str"  # Default, could be inferred better
                    if arg.default_value and isinstance(arg.default_value, list):
                        param_def["options"] = arg.default_value
                else:  # STRING or JSON
                    param_def["type"] = "string"
                    param_def["data_type"] = "str"
                
                schema["parameters"][param_name] = param_def
        
        return schema
    
    def export_defaults_yaml(self, output_path: str, sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Export vLLM CLI defaults to a YAML file for use in configuration loading.
        
        Args:
            output_path: Path to save the defaults.yaml file
            sections: List of sections to include. If None, includes all optimization-relevant sections.
        
        Returns:
            Dictionary of exported defaults
        """
        if sections is None:
            # Default to optimization-relevant sections
            sections = ['cacheconfig', 'schedulerconfig', 'modelconfig', 'parallelconfig']
        
        # Parse arguments if not already done
        if not self.arguments:
            self.parse()
            
        version = self.get_vllm_version()
        
        defaults = {
            "# vLLM CLI defaults extracted automatically": None,
            "# vLLM version: " + version: None,
            "# Generated sections: " + ", ".join(sections): None,
            "version": version,
            "defaults": {}
        }
        
        for section in sections:
            section_defaults = {}
            for arg in self.get_arguments_by_section(section):
                # Skip positional arguments and help
                if not arg.long_name.startswith('--') or arg.long_name in ['--help', '--config']:
                    continue
                
                # Convert --arg-name to arg_name
                param_name = arg.long_name[2:].replace('-', '_')
                
                # Only include parameters that have meaningful defaults
                if arg.default_value is not None:
                    section_defaults[param_name] = arg.default_value
            
            if section_defaults:
                defaults["defaults"][section] = section_defaults
        
        # Clean up None comments for YAML output
        clean_defaults = {k: v for k, v in defaults.items() if v is not None}
        
        # Save to file
        import yaml
        with open(output_path, 'w') as f:
            # Write comments manually since PyYAML doesn't handle None values well
            f.write("# vLLM CLI defaults extracted automatically\n")
            f.write(f"# vLLM version: {version}\n")
            f.write(f"# Generated sections: {', '.join(sections)}\n")
            f.write("# This file contains default values from vLLM CLI arguments\n")
            f.write("# Use this instead of hardcoding defaults in Python code\n\n")
            yaml.dump(clean_defaults, f, default_flow_style=False, sort_keys=True)
        
        return clean_defaults
    
    def export_versioned_defaults(self, base_dir: str, sections: Optional[List[str]] = None) -> str:
        """
        Export vLLM CLI defaults to a versioned file in the specified directory.
        
        Args:
            base_dir: Base directory for versioned defaults (e.g., 'auto_tune_vllm/schemas/vllm_defaults')
            sections: List of sections to include. If None, includes all optimization-relevant sections.
        
        Returns:
            Path to the created version-specific defaults file
        """
        from pathlib import Path
        
        # Parse arguments if not already done
        if not self.arguments:
            self.parse()
            
        version = self.get_vllm_version()
        safe_version = version.replace('.', '_')  # Make filesystem-safe
        
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Create version-specific file
        version_file = base_path / f"v{safe_version}.yaml"
        
        # Export defaults
        self.export_defaults_yaml(str(version_file), sections)
        
        # Also create or update a 'latest.yaml' symlink/copy
        latest_file = base_path / "latest.yaml"
        if latest_file.exists():
            latest_file.unlink()
        
        # Copy the file instead of symlink for better cross-platform compatibility
        import shutil
        shutil.copy2(str(version_file), str(latest_file))
        
        return str(version_file)
    
    def get_parameter_defaults(self, param_name: str) -> Any:
        """
        Get the default value for a specific parameter.
        
        Args:
            param_name: Parameter name (with or without -- prefix)
        
        Returns:
            Default value or None if not found
        """
        # Ensure the CLI has been parsed so self.arguments is populated
        if not self.arguments:
            self.parse()
         # Normalize parameter name
        if not param_name.startswith('--'):
            param_name = '--' + param_name.replace('_', '-')
        
        if param_name in self.arguments:
            return self.arguments[param_name].default_value
        
        return None


def main():
    """Demo usage of the parser."""
    parser = VLLMCLIParser(venv_path=".venv")
    
    try:
        print("Parsing vLLM CLI arguments...")
        arguments = parser.parse()
        
        print(f"\nFound {len(arguments)} arguments across {len(parser.get_all_sections())} sections:")
        for section in parser.get_all_sections():
            args_in_section = parser.get_arguments_by_section(section)
            print(f"  {section}: {len(args_in_section)} arguments")
        
        # Show some examples
        print("\nExample arguments:")
        for i, (name, arg) in enumerate(arguments.items()):
            if i >= 5:  # Show first 5
                break
            print(f"  {name}: {arg.arg_type.value}, default={arg.default_value}")
        
        # Generate schema for specific sections
        important_sections = ['cacheconfig', 'schedulerconfig', 'modelconfig']
        schema = parser.generate_parameter_schema(important_sections)
        
        print(f"\nGenerated schema with {len(schema['parameters'])} parameters")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
