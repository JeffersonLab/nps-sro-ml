# Contributing code

The one rule of contributing is to have empathy to your collaborators ! Think of other contributor when you write code (Just imagine someone pushed thousand lines of code without formating, documentation and contain repeated logics and you have to spends weeks just to understand, refactor and integrate. ) Try to keep code changes focused, readable, and reproducible. A reviewer should be able to understand both the software change and its effect on the analysis.

## Commit checklist

Before committing your code, be sure to check the following list 

-[] add documentation in complicated functions. 
-[] add or update unittests when behavior changes or a bug is fixed.
- double-checked no generated files, build directories, large datasets, ROOT files, and model checkpoints, etc ... are committed.
- run the formatter

## Python

### Format

The repository use [`black`]() formatter to organize python codes. The formatter has a simple configuration to sets the maximum preferred line, preserves the original quotation style used in string literals, and keeps the default magic trailing comma. Specific configuration can be found under `tool.black` in `pyproject.toml`. Before you commit, run
```bash
uv run black ${location}
```

[Optional] Comments are formatted using [`ruff`]() where the configuration can be found under [tool.ruff] in `pyproject.toml`. Before you commit, run
```bash
uv run ruff check ${location}
```
This generates a message to ...


### Unit Tests [Optional]

During development or committing to a debug task, you are highly recommended to run unit tests to demonstrate the usage in general and also edge cases. THe repository uses [`pytest`]() suite. Befeore you commit, run 

```bash
uv run pytest ${location}
```

Run the full test suite before requesting final review when your environment
supports it. 

### Docstrings and comments

Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code. Public or
complicated functions, classes, and methods should have docstrings that explain
their purpose, parameters, return values, and important assumptions or raised
exceptions. Docstring conventions are formally described by
[PEP 257](https://peps.python.org/pep-0257/).

This repository configures Ruff to use the
[NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html).
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

Use comments to explain *why* an implementation or physics choice is necessary,
not to restate what a line of code already says.

## C++

The converter uses C++17 and the repository's `.clang-format` configuration.
Run the provided formatter from the repository root before committing C++
changes:

```bash
cd converter
./format.sh
cd ..
```

Alternatively, format an individual file from the repository root:

```bash
clang-format -i --style=file converter/sources/NPS.cpp
```

Inspect the formatting diff before committing. Do not mix a repository-wide
formatting change with a functional change.

When your environment has the required ROOT and LibTorch dependencies, build
the converter as described in the [converter instructions](../instruction-converter.md):

```bash
cmake -S converter -B converter/build
cmake --build converter/build -j 8
```

If those dependencies are unavailable, say so in the PR description and ask a
reviewer who has the appropriate environment to verify the build.

## Scientific and machine-learning changes

For changes that can affect scientific conclusions or model performance, make
the comparison reviewable:

- identify the dataset or data-taking period without committing restricted or
  large raw data;
- document preprocessing, selections, feature definitions, targets, units, and
  coordinate conventions;
- preserve train/validation/test separation and check for data leakage;
- report the random seed and important hyperparameters;
- compare against the current baseline with an appropriate metric and, when
  useful, uncertainty or variation across seeds; and
- save enough configuration and provenance to reproduce the result.

If a change intentionally alters a physics definition or expected numerical
result, highlight that fact near the top of the PR description.

