# analysis/

Jupyter notebooks that drive `simcore` through the `orbitgnc` Python module (`bindings/`).

## Setup

```sh
python3 -m venv analysis/.venv                         # the presets look here for Python
analysis/.venv/bin/pip install -r analysis/requirements.txt
cmake --preset release && cmake --build --preset release
analysis/.venv/bin/jupyter lab analysis/
```

The notebooks find the module in `build/release/python` (or `build/debug/python`). Configure after
creating the venv: CMake looks for Python (with NumPy) when it configures, and skips the module
with a message if there isn't one.

`requirements-test.txt` is the subset the module's tests need (CI installs only that);
`requirements.txt` adds Matplotlib and JupyterLab. Direct dependencies are pinned; theirs are not.

## Notebooks

| Notebook | What it shows |
|---|---|
| `01_orbit_basics.ipynb` | J2 nodal regression of a 420 km, 51.6° orbit over a day vs the first-order secular rate |

Notebooks are committed with their outputs so GitHub renders the plots. Re-run one in place with
`analysis/.venv/bin/jupyter nbconvert --to notebook --execute --inplace analysis/<name>.ipynb`.
