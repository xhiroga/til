"""
Build hook that transforms relative imports to absolute imports using temporary files
"""

import ast
import tempfile
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class RelativeToAbsoluteTransformer(ast.NodeTransformer):
    def __init__(self, package_name: str):
        self.package_name = package_name

    def visit_ImportFrom(self, node):
        if node.level and node.level > 0:
            abs_module = f"{self.package_name}.{node.module}" if node.module else self.package_name
            print(f"[TRANSFORM] from {'.' * node.level}{node.module or ''} -> from {abs_module}")
            return ast.ImportFrom(module=abs_module, names=node.names, level=0)
        return node


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        print(f"[BUILD] Starting import transformation for version {version}")
        
        src_path = Path(self.root) / "src" / "sandbox"
        if not src_path.exists():
            return

        files_to_transform = [f for f in src_path.rglob("*.py") if self._has_relative_imports(f)]
        if not files_to_transform:
            print("[BUILD] No files with relative imports found")
            return

        print(f"[BUILD] Transforming {len(files_to_transform)} files")
        temp_dir = tempfile.mkdtemp(prefix="hatch_build_")
        transformer = RelativeToAbsoluteTransformer("sandbox")
        
        for py_file in files_to_transform:
            self._transform_file(py_file, transformer, temp_dir, src_path, build_data)

    def _has_relative_imports(self, file_path: Path) -> bool:
        try:
            return "from ." in file_path.read_text(encoding="utf-8")
        except Exception:
            return False

    def _transform_file(self, file_path: Path, transformer, temp_dir: str, src_base: Path, build_data: dict):
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        new_tree = transformer.visit(tree)
        new_content = ast.unparse(new_tree)
        
        rel_path = file_path.relative_to(src_base)
        temp_file_path = Path(temp_dir) / rel_path
        temp_file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file_path.write_text(new_content, encoding="utf-8")
        
        build_data.setdefault('force_include', {})[str(temp_file_path)] = f"sandbox/{rel_path}"
        print(f"[BUILD] Transformed {file_path.name}")

    def finalize(self, version, build_data, artifact_path):
        print(f"[BUILD] Transformation completed - {artifact_path}")
        print("[BUILD] Source files remain unchanged")
