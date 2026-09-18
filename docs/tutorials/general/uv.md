# Using uv

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project
manager. It can install Python, create virtual environments, manage
dependencies, produce a lockfile, and run commands in a reproducible project
environment.

## Install uv

::: tip JLab users
Install `uv` from a terminal opened inside a
[JLab VDI session](./vdi.md), rather than from an SSH session on ifarm. The VDI
session is already inside the Jefferson Lab network, which makes it the
preferred environment for this interactive setup.
:::

On Linux and macOS, use the standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatively, install it in an isolated Python environment:

```bash
pipx install uv
```

Check the installation with `uv --version`. See the
[official installation guide](https://docs.astral.sh/uv/getting-started/installation/)
for package-manager and version-pinned installation options.

## Start a project

Create a new project:

```bash
uv init my-project
cd my-project
```

Or initialize an existing directory:

```bash
uv init
```

The main project files are:

| Path | Purpose |
| --- | --- |
| `pyproject.toml` | Project metadata, Python requirement, and direct dependencies |
| `uv.lock` | Exact, cross-platform dependency resolution |
| `.python-version` | Preferred Python version |
| `.venv/` | Local virtual environment; do not commit it |

Commit `pyproject.toml` and `uv.lock` to version control. Do not edit
`uv.lock` manually.

## Select Python

`uv` can use an existing interpreter or download one when needed:

```bash
uv python list
uv python install 3.12
uv python pin 3.12
```

`uv python pin` writes the selected version to `.python-version`. A particular
interpreter can also be requested for one command:

```bash
uv venv --python 3.12
```

## Manage dependencies

Add and remove runtime dependencies:

```bash
uv add requests
uv add "numpy>=2,<3"
uv remove requests
```

Add development dependencies:

```bash
uv add --dev pytest ruff
```

Add a dependency to a named group or optional extra:

```bash
uv add --group docs mkdocs
uv add --optional plotting matplotlib
```

These commands update `pyproject.toml`, resolve `uv.lock`, and synchronize the
environment. Inspect the resolved dependency graph with:

```bash
uv tree
```

## Synchronize an existing project

After cloning a project, create or update its `.venv` from the lockfile:

```bash
uv sync
```

Select optional dependencies or groups when needed:

```bash
uv sync --extra plotting
uv sync --all-extras
uv sync --group docs
uv sync --no-dev
```

By default, synchronization is exact and may remove packages that are not
declared by the project. Use normal dependency commands instead of manually
installing packages into the managed environment.

Useful lockfile checks for CI are:

```bash
uv lock --check
uv sync --locked
```

`--locked` fails if `pyproject.toml` and `uv.lock` disagree. `--frozen` uses
the existing lockfile without checking whether it is current.

## Run commands

`uv run` ensures the project environment is ready, then runs a command inside
it:

```bash
uv run python app.py
uv run pytest
uv run python -c "import sys; print(sys.executable)"
```

Activating `.venv` is optional. If an interactive activated shell is more
convenient:

```bash
source .venv/bin/activate
```

On Windows PowerShell, use:

```powershell
.venv\Scripts\Activate.ps1
```

Add a temporary dependency for one invocation without changing the project:

```bash
uv run --with rich python example.py
```

## Run Python tools

`uvx` runs a command-line tool in a temporary isolated environment:

```bash
uvx ruff check .
uvx black --check .
```

It is an alias for `uv tool run`. Install a frequently used tool globally with:

```bash
uv tool install ruff
uv tool list
uv tool upgrade --all
```

Project-specific tools such as test runners and formatters usually belong in a
development dependency group instead.

## Use the pip-compatible interface

For projects that do not use `pyproject.toml` and `uv.lock`, `uv` also
provides pip-compatible commands:

```bash
uv venv
uv pip install -r requirements.txt
uv pip compile requirements.in -o requirements.txt
uv pip sync requirements.txt
```

The project workflow (`uv add`, `uv sync`, and `uv run`) is preferable for
new projects. The `uv pip` interface is useful when maintaining an existing
`requirements.txt` workflow.

## Common workflow

```bash
# First-time setup
uv sync

# Add a package
uv add pandas

# Run code and tests
uv run python app.py
uv run pytest

# Before committing
uv lock --check
```

Use `uv help <command>` for local command help, or consult the
[official uv documentation](https://docs.astral.sh/uv/) for advanced topics
such as workspaces, private package indexes, publishing, and dependency
sources.


