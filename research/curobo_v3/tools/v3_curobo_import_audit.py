import argparse
import ast
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check cuRobo symbols directly imported by our research tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    imports = set()
    for path in Path(__file__).resolve().parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("curobo."):
                imports.update((node.module, alias.name) for alias in node.names if alias.name != "*")
    errors = []
    for module, name in sorted(imports):
        try:
            getattr(importlib.import_module(module), name)
        except Exception as error:
            errors.append({"module": module, "symbol": name, "error": str(error)})
    result = {"import_pairs_checked": len(imports), "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
