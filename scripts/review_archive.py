"""Static inventory of every Python source file; does not import legacy frameworks."""

import ast
from collections import defaultdict
import json
from pathlib import Path
import re
import warnings

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "numpy_int_alias": r"\bnp\.int\b",
    "legacy_sampling_api": r"\.fit_sample\(",
    "tensorflow_contrib": r"\btf\.contrib\b|\btensorflow\.contrib\b",
    "full_dataset_fit_candidate": r"\.fit(?:_transform)?\(X(?:_resampled)?,\s*y(?:_resampled)?\)",
}


def scan():
    files, errors, patterns = [], [], {name: [] for name in PATTERNS}
    functions = defaultdict(list)
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".git", ".preview", ".venv", "venv", "__pycache__", "node_modules"} for part in path.relative_to(ROOT).parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        source = path.read_text(encoding="utf-8-sig")
        files.append({"path": relative, "lines": len(source.splitlines())})
        for name, pattern in PATTERNS.items():
            for number, line in enumerate(source.splitlines(), 1):
                if not line.lstrip().startswith("#") and re.search(pattern, line):
                    patterns[name].append({"path": relative, "line": number})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source, filename=relative)
        except SyntaxError as exc:
            errors.append({"path": relative, "line": exc.lineno, "error": exc.msg})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.end_lineno - node.lineno >= 10:
                functions[ast.dump(node, include_attributes=False)].append({"path": relative, "line": node.lineno, "function": node.name})
    report = {"scope": "All Python files parsed; pattern hits are review leads, not runtime proof.",
              "files": files, "syntax_errors": errors, "legacy_patterns": patterns,
              "remaining_exact_function_duplicates": [locations for locations in functions.values() if len(locations) > 1]}
    output = ROOT / "docs" / "archive-review.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Parsed {len(files)} Python files; syntax errors: {len(errors)}")
    print("Legacy patterns:", {name: len(hits) for name, hits in patterns.items()})
    print(f"Remaining substantial exact duplicate groups: {len(report['remaining_exact_function_duplicates'])} (mostly historical model examples)")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    scan()
