"""
Version management utilities for vLLM defaults.

This module provides functionality to:
1. List available vLLM defaults versions
2. Load version-specific defaults
3. Compare defaults across versions
4. Manage version-specific configuration loading
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re


@dataclass
class VLLMDefaultsVersion:
    """Represents a version of vLLM defaults."""
    version: str
    file_path: Path
    creation_time: datetime
    sections: List[str]
    total_defaults: int
    
    def __str__(self) -> str:
        return f"vLLM {self.version} ({self.total_defaults} defaults)"


class VLLMVersionManager:
    """Manages versioned vLLM defaults files."""
    
    def __init__(self, defaults_dir: Optional[str] = None):
        """
        Initialize the version manager.
        
        Args:
            defaults_dir: Directory containing versioned defaults. 
                         Defaults to auto_tune_vllm/schemas/vllm_defaults
        """
        if defaults_dir is None:
            # Use package default location
            defaults_dir = Path(__file__).parent.parent / "schemas" / "vllm_defaults"
        
        self.defaults_dir = Path(defaults_dir)
        self.defaults_dir.mkdir(parents=True, exist_ok=True)
    
    def list_available_versions(self) -> List[VLLMDefaultsVersion]:
        """List all available vLLM defaults versions."""
        versions = []
        
        # Pattern to match version files (v0_10_1_1.yaml)
        version_pattern = re.compile(r'^v(\d+(?:_\d+)*).yaml$')
        
        for file_path in self.defaults_dir.glob("v*.yaml"):
            if file_path.name == "latest.yaml":  # Skip the latest symlink
                continue
            
            match = version_pattern.match(file_path.name)
            if not match:
                continue
            
            # Convert back to standard version format (0.10.1.1)
            version_str = match.group(1).replace('_', '.')
            
            try:
                # Load the file to get metadata
                with open(file_path) as f:
                    data = yaml.safe_load(f)
                
                defaults_data = data.get('defaults', {})
                sections = list(defaults_data.keys())
                total_defaults = sum(len(section_defaults) for section_defaults in defaults_data.values())
                
                creation_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                versions.append(VLLMDefaultsVersion(
                    version=version_str,
                    file_path=file_path,
                    creation_time=creation_time,
                    sections=sections,
                    total_defaults=total_defaults
                ))
                
            except Exception as e:
                print(f"Warning: Could not load version file {file_path}: {e}")
                continue
        
        # Sort by version (newest first)
        versions.sort(key=lambda v: self._version_sort_key(v.version), reverse=True)
        return versions
    
    def _version_sort_key(self, version: str) -> Tuple[int, ...]:
        """Create a sort key for version strings."""
        try:
            return tuple(int(x) for x in version.split('.'))
        except ValueError:
            return (0,)  # Fallback for invalid versions
    
    def get_latest_version(self) -> Optional[VLLMDefaultsVersion]:
        """Get the latest available version."""
        versions = self.list_available_versions()
        return versions[0] if versions else None
    
    def get_version(self, version: str) -> Optional[VLLMDefaultsVersion]:
        """Get a specific version."""
        for v in self.list_available_versions():
            if v.version == version:
                return v
        return None
    
    def load_defaults(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load defaults for a specific version.
        
        Args:
            version: Version string (e.g., "0.10.1.1"). If None, loads latest.
        
        Returns:
            Dictionary of defaults data
        """
        if version is None:
            # Use latest.yaml if available, otherwise latest version
            latest_file = self.defaults_dir / "latest.yaml"
            if latest_file.exists():
                with open(latest_file) as f:
                    return yaml.safe_load(f)
            else:
                latest_version = self.get_latest_version()
                if latest_version is None:
                    raise ValueError("No vLLM defaults versions available")
                version = latest_version.version
        
        version_info = self.get_version(version)
        if version_info is None:
            raise ValueError(f"vLLM defaults version {version} not found")
        
        with open(version_info.file_path) as f:
            return yaml.safe_load(f)
    
    def get_defaults_path(self, version: Optional[str] = None) -> Path:
        """
        Get the file path for a specific version's defaults.
        
        Args:
            version: Version string. If None, returns path to latest.
        
        Returns:
            Path to the defaults file
        """
        if version is None:
            return self.defaults_dir / "latest.yaml"
        
        version_info = self.get_version(version)
        if version_info is None:
            raise ValueError(f"vLLM defaults version {version} not found")
        
        return version_info.file_path
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare defaults between two versions.
        
        Args:
            version1: First version to compare
            version2: Second version to compare
        
        Returns:
            Dictionary with comparison results
        """
        defaults1 = self.load_defaults(version1)
        defaults2 = self.load_defaults(version2)
        
        def flatten_defaults(defaults_data):
            """Flatten nested defaults for comparison."""
            flat = {}
            for section, section_defaults in defaults_data.get('defaults', {}).items():
                for param, value in section_defaults.items():
                    flat[f"{section}.{param}"] = value
            return flat
        
        flat1 = flatten_defaults(defaults1)
        flat2 = flatten_defaults(defaults2)
        
        all_keys = set(flat1.keys()) | set(flat2.keys())
        
        added = []
        removed = []
        changed = []
        unchanged = []
        
        for key in all_keys:
            if key not in flat1:
                added.append(key)
            elif key not in flat2:
                removed.append(key)
            elif flat1[key] != flat2[key]:
                changed.append({
                    'parameter': key,
                    'old_value': flat1[key],
                    'new_value': flat2[key]
                })
            else:
                unchanged.append(key)
        
        return {
            'version1': version1,
            'version2': version2,
            'added': sorted(added),
            'removed': sorted(removed),
            'changed': changed,
            'unchanged': sorted(unchanged),
            'summary': {
                'total_v1': len(flat1),
                'total_v2': len(flat2),
                'added_count': len(added),
                'removed_count': len(removed),
                'changed_count': len(changed),
                'unchanged_count': len(unchanged)
            }
        }
    
    def cleanup_old_versions(self, keep_count: int = 5) -> List[str]:
        """
        Remove old version files, keeping only the most recent ones.
        
        Args:
            keep_count: Number of versions to keep
        
        Returns:
            List of deleted version strings
        """
        versions = self.list_available_versions()
        
        if len(versions) <= keep_count:
            return []
        
        # Keep the most recent versions
        to_delete = versions[keep_count:]
        deleted = []
        
        for version_info in to_delete:
            try:
                version_info.file_path.unlink()
                deleted.append(version_info.version)
            except Exception as e:
                print(f"Warning: Could not delete {version_info.file_path}: {e}")
        
        return deleted
    
    def get_version_info_summary(self) -> str:
        """Get a summary of available versions."""
        versions = self.list_available_versions()
        
        if not versions:
            return "No vLLM defaults versions available."
        
        lines = [f"Available vLLM defaults versions ({len(versions)} total):\n"]
        
        for i, version in enumerate(versions):
            marker = "→ " if i == 0 else "  "
            lines.append(f"{marker}{version}")
            lines.append(f"   Created: {version.creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"   Sections: {', '.join(version.sections)}")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Demo the version manager functionality."""
    print("🗂️  vLLM Version Manager Demo")
    print("=" * 50)
    
    manager = VLLMVersionManager()
    
    print("📋 Available versions:")
    print(manager.get_version_info_summary())
    
    latest = manager.get_latest_version()
    if latest:
        print(f"📌 Latest version: {latest}")
        
        # Show some defaults from latest version
        try:
            defaults = manager.load_defaults()
            print("\n🔍 Sample defaults from latest version:")
            for section, section_defaults in list(defaults.get('defaults', {}).items())[:2]:
                print(f"   {section}:")
                for param, value in list(section_defaults.items())[:3]:
                    print(f"     {param} = {value}")
        except Exception as e:
            print(f"   Could not load defaults: {e}")


if __name__ == "__main__":
    main()
