# 🤝 Contributing Code

The golden rule of contributing is to have ***empathy for your collaborators***.

Code is rarely written for just one person. Your collaborators may need to review it, debug it, extend it, or understand it months after it was written. A large contribution with inconsistent formatting, little documentation, duplicated logic, or unclear structure can make that work unnecessarily difficult.

Before committing, take a moment to consider the person who will read your code next. This guide outlines a few practices to follow before submitting your changes. The goal is not perfection, but to keep the codebase clear, maintainable, and easy for everyone to work with.

## ✅ Commit checklist

Before committing your changes, please make sure that you have:

* 🎨 Run the formatter and ensure the code follows the repository's style guidelines.
* 📝 Add documentation or comments for non-trivial functions and logic.
* ✅ Add or update unit tests when behavior changes or a bug is fixed.
* 🙈 Make sure unnecessary files (e.g. build directories, generated files, API keys, etc.) are not committed.

## 🐍 Python

### 🎨 Format

* The repository uses [`black`](https://github.com/psf/black) as the Python formatter. Before you commit, run

```bash
uv run black ${workdir}
```

This command modifies all Python files in `${workdir}`. Alternatively, format a single file by running `uv run black ${file}`. Specific formatting configuration can be found under `tool.black` in `pyproject.toml`.

* [Optional] Comments and docstrings are checked and fixed using [`ruff`](https://github.com/astral-sh/ruff). Before you commit, run

```bash
uv run ruff check --fix ${workdir}
```

This applies configured Ruff fixes to files under `${workdir}`. Specific configuration can be found under `tool.ruff` in `pyproject.toml`.

### 📝 Documentation

Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code. Public or complicated functions, classes, and methods should have docstrings that explain their purpose, parameters, return values, important assumptions, and raised exceptions.

Docstring conventions are formally described by [PEP 257](https://peps.python.org/pep-0257/). This repository configures Ruff to use the [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html).

For example:

```python
def calibrated_energy(adc, gain):
    """Convert ADC values to calibrated energies.

    Parameters
    ----------
    adc : numpy.ndarray
        Raw ADC values with shape ``(n_hits,)``.
    gain : float
        Calibration gain in GeV per ADC count.

    Returns
    -------
    numpy.ndarray
        Calibrated energies in GeV with shape ``(n_hits,)``.

    Raises
    ------
    ValueError
        If ``gain`` is not positive.
    """
```

Use comments to explain ***why*** an implementation or physics choice is necessary, rather than restating what a line of code already says.

### ✅ Unit Tests

Unit tests are completely optional. That said, it is not difficult to generate unit tests with AI. During development, or when committing a bug fix, you are highly encouraged to run unit tests to demonstrate the robustness of the code.

The repository uses [`pytest`](https://docs.pytest.org/en/stable/) to organize the [`tests`](https://github.com/JeffersonLab/nps-sro-ml/tree/main/pytorch_src/tests) suite.

```bash
uv run pytest ${workdir}
```

Alternatively, replace `${workdir}` with the changed test files or relevant test paths.

## ⚡ C++

### 🎨 Format

This repository includes C++ code in [`converter`](https://github.com/JeffersonLab/nps-sro-ml/tree/main/converter) for generating training data. Before committing C++ changes, run the provided formatter:

```bash
./format.sh
```

Alternatively, format an individual file from the repository root:

```bash
clang-format -i --style=file converter/sources/NPS.cpp
```

Inspect the formatting diff before committing. Do not mix a repository-wide formatting change with a functional change.

### 📝 Documentation

There is currently no strict rule for documenting C++ source code. You are encouraged to add short comments for complicated logic and follow [Doxygen](https://www.doxygen.nl/) conventions for documenting classes and public interfaces.

For example:

```cpp
/**
 * @brief Converts raw detector data into reconstructed NPS events.
 *
 * Handles event decoding, calibration, and preparation of the output
 * used for downstream training-data generation.
 */
class MyClass {
    ...
};
```

Use comments primarily to explain non-obvious implementation decisions, assumptions, and physics-related choices.
