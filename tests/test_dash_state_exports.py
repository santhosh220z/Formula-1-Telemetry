import ast
import pathlib

import dash_state

_ENTRY_FILES = (
    pathlib.Path("app.py"),
    pathlib.Path("pages/1_Lap Replay.py"),
)


def _dash_state_imports(path: pathlib.Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "dash_state":
            names.update(alias.name for alias in node.names)
    return names


def test_every_dash_state_import_exists():
    missing = {}
    for path in _ENTRY_FILES:
        missing[path.name] = sorted(
            name for name in _dash_state_imports(path) if not hasattr(dash_state, name)
        )
        assert not missing[path.name], missing[path.name]


def test_dash_state_names_are_importable():
    for name in sorted(_dash_state_imports(_ENTRY_FILES[0])):
        assert hasattr(dash_state, name), name
