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

To run it from PyCharm on the open file, add an external tool in Settings > Tools > External Tools:
- Program: `$PyInterpreterDirectory$\ruff.exe`
- Arguments: `check --config <path to LNCM-lib>\pyproject.toml "$FilePath$"`. The explicit config path makes the same rules apply to files in other repos (e.g. 2p_code) that have no ruff config.
- Working directory: `$FileDir$`

Then use Tools > External Tools > ruff (or assign a shortcut in Keymap). `--fix` removes unused imports and similar automatically; check the diff afterwards.
