"""
Build hook that transforms relative imports to absolute imports using temporary files
"""

import ast
import tempfile
import shutil
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class RelativeToAbsoluteTransformer(ast.NodeTransformer):
    """Transform relative imports to absolute imports."""

    def __init__(self, package_name: str):
        self.package_name = package_name

    def visit_ImportFrom(self, node):
        """Transform relative from imports to absolute imports."""
        if node.level and node.level > 0:  # Relative import
            if node.level == 1:  # from .module
                if node.module:
                    # from .utils import func -> from sandbox.utils import func
                    abs_module = f"{self.package_name}.{node.module}"
                else:
                    # from . import module -> from sandbox import module
                    abs_module = self.package_name
            elif node.level == 2:  # from ..module
                if node.module:
                    # from ..utils import func -> from sandbox.utils import func
                    abs_module = f"{self.package_name}.{node.module}"
                else:
                    # from .. import module -> from sandbox import module
                    abs_module = self.package_name
            else:
                # Handle deeper levels if needed
                abs_module = self.package_name

            print(
                f"[TRANSFORM] from {'.' * node.level}{node.module or ''} -> from {abs_module}"
            )

            return ast.ImportFrom(
                module=abs_module,
                names=node.names,
                level=0,  # Make it absolute
            )
        return node


class CustomBuildHook(BuildHookInterface):
    """Build hook that transforms relative imports to absolute imports without modifying source"""

    def initialize(self, version, build_data):
        """Transform Python files in temporary directory and include in build."""
        print(
            f"[BUILD] Starting non-destructive relative->absolute import transformation for version {version}"
        )

        # Find all Python files in src/sandbox
        src_path = Path(self.root) / "src" / "sandbox"
        if not src_path.exists():
            print(f"[BUILD] No src/sandbox directory found")
            return

        python_files = list(src_path.rglob("*.py"))
        transformed_files = []
        
        # Process files that need transformation
        for py_file in python_files:
            if self._needs_transformation(py_file):
                transformed_files.append(py_file)

        if not transformed_files:
            print(f"[BUILD] No files with relative imports found")
            return

        print(f"[BUILD] Found {len(transformed_files)} files with relative imports to transform")

        # Create temporary directory for transformed files
        temp_dir = tempfile.mkdtemp(prefix="hatch_build_")
        print(f"[BUILD] Using temporary directory: {temp_dir}")
        
        try:
            transformer = RelativeToAbsoluteTransformer("sandbox")
            
            for py_file in transformed_files:
                self._transform_file_to_temp(py_file, transformer, temp_dir, src_path, build_data)
                
        except Exception as e:
            print(f"[BUILD] Error during transformation: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _needs_transformation(self, file_path: Path) -> bool:
        """Check if file contains relative imports."""
        try:
            content = file_path.read_text(encoding="utf-8")
            return "from ." in content
        except Exception:
            return False

    def _transform_file_to_temp(
        self, file_path: Path, transformer: RelativeToAbsoluteTransformer, 
        temp_dir: str, src_base: Path, build_data: dict
    ):
        """Transform file and save to temporary directory, then add to build data."""
        try:
            # Read original content
            content = file_path.read_text(encoding="utf-8")

            # Parse and transform
            tree = ast.parse(content)
            new_tree = transformer.visit(tree)

            # Convert back to code
            new_content = ast.unparse(new_tree)

            # Calculate relative path from src/sandbox
            rel_path = file_path.relative_to(src_base)
            
            # Create temp file with same relative structure
            temp_file_path = Path(temp_dir) / rel_path
            temp_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write transformed content to temp file
            temp_file_path.write_text(new_content, encoding="utf-8")
            
            # Add to build data - map temp file to distribution path
            dist_path = f"sandbox/{rel_path}"
            
            # Initialize force_include if not exists
            if 'force_include' not in build_data:
                build_data['force_include'] = {}
                
            build_data['force_include'][str(temp_file_path)] = dist_path
            
            print(f"[BUILD] Transformed {file_path.name} -> temp file -> {dist_path}")

        except Exception as e:
            print(f"[BUILD] Error transforming {file_path}: {e}")

    def finalize(self, version, build_data, artifact_path):
        """Build completion."""
        print(f"[BUILD] Non-destructive import transformation completed - artifact: {artifact_path}")
        print(f"[BUILD] Source files remain unchanged")
