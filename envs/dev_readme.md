# Developing LNCM-lib

## Changing packages
To change packages, edit `pyproject.toml` and run `uv lock`, then commit both files.

Qt: PySide6 is the only Qt binding in the env. Don't add another one (e.g. PyQt6 via `suite2p[gui]`): bindings ship Qt DLLs with the same names, and when one binding's DLLs are loaded first, importing the other fails with `DLL load failed ... The specified procedure could not be found`.

## Conventions for paths
- Library classes store paths as `pathlib.Path` and accept `str` or `Path` as input.
- Join folders with `/` on `Path` objects. Don't use string `+`, hard-coded separators or trailing slashes.
- Each class builds its own file names in one method (e.g. `ImagingSession.get_file_with_suffix`).
- Library code does not call `os.chdir`.
- `touch_path` returns a string with a trailing slash, for use in analysis scripts.
- GUIs may keep their own folder strings. `_Dependencies` is not converted.

## Linting with ruff
ruff is installed in the env (`dev` group in `pyproject.toml`) as `<uv_env_dir>\<username>\lncm\Scripts\ruff.exe`. The config in `pyproject.toml` only reports likely bugs (unused or undefined names, syntax errors), not style. Run it on the files you changed, not on the whole repo.

In the Claude Code container, a hook (`/workspace/.claude/hooks/ruff-before-commit.sh`) blocks commits of files with ruff errors, including errors that were there before. When you start editing a file that has ruff errors, fix them first and commit that as its own commit (e.g. "Fix ruff errors in <file>"), then make the actual change. That way the real change stays a single commit and isn't blocked later.

Star imports (`from x import *`) are not allowed in LNCM-lib: they hide undefined names from ruff, and they make every name a module imports part of what scripts depend on. Analysis scripts may use them: 2p_code's `ruff.toml` extends this config and allows them, so scripts can keep `from _LNCM_Analysis import *`. Another repo can do the same with its own `ruff.toml`:
```
extend = "../LNCM-lib/pyproject.toml"

[lint]
ignore = ["F403", "F405"]
```

To run it from PyCharm on the open file, add an external tool in Settings > Tools > External Tools:
- Program: `$PyInterpreterDirectory$\ruff.exe`
- Arguments: `check "$FilePath$"`. ruff uses the config of the repo the file is in. A repo without a ruff config gets ruff's default rules, so give it a `ruff.toml` like the one above.
- Working directory: `$FileDir$`

Then use Tools > External Tools > ruff (or assign a shortcut in Keymap). `--fix` removes unused imports and similar automatically; check the diff afterwards.
