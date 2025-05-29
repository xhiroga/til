"""
Build hook that transforms relative imports to absolute imports
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
            
            print(f"[TRANSFORM] from {'.' * node.level}{node.module or ''} -> from {abs_module}")
            
            return ast.ImportFrom(
                module=abs_module,
                names=node.names,
                level=0  # Make it absolute
            )
        return node


class CustomBuildHook(BuildHookInterface):
    """Build hook that transforms relative imports to absolute imports"""
    
    def initialize(self, version, build_data):
        """Transform Python files during build."""
        print(f"[BUILD] Starting relative->absolute import transformation for version {version}")
        
        # Find all Python files in src/sandbox
        src_path = Path(self.root) / "src" / "sandbox"
        if not src_path.exists():
            print(f"[BUILD] No src/sandbox directory found")
            return
        
        python_files = list(src_path.rglob("*.py"))
        print(f"[BUILD] Found {len(python_files)} Python files to process")
        
        transformer = RelativeToAbsoluteTransformer("sandbox")
        
        for py_file in python_files:
            self._transform_file(py_file, transformer)
    
    def _transform_file(self, file_path: Path, transformer: RelativeToAbsoluteTransformer):
        """Transform imports in a single Python file."""
        try:
            # Read original content
            content = file_path.read_text(encoding='utf-8')
            
            # Check if file has relative imports
            if not ('from .' in content):
                print(f"[BUILD] No relative imports in {file_path.name}")
                return
            
            # Parse and transform
            tree = ast.parse(content)
            new_tree = transformer.visit(tree)
            
            # Convert back to code
            import astor
            new_content = astor.to_source(new_tree)
            
            # Write transformed content back to the same file
            file_path.write_text(new_content, encoding='utf-8')
            print(f"[BUILD] Transformed: {file_path.relative_to(self.root)}")
            
        except Exception as e:
            print(f"[BUILD] Error transforming {file_path}: {e}")
    
    def finalize(self, version, build_data, artifact_path):
        """Build completion."""
        print(f"[BUILD] Import transformation completed - artifact: {artifact_path}")
