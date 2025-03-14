# stressgait_analysis




## Project structure

```
stressgait_analysis
│   README.md
├── src
|   ├── stressgait_analysis  # The main library for the project
|
├── experiments  # The main folder for all experiments. Each experiment has its own subfolder
|
|   pyproject.toml  # The required python dependencies for the project
|   uv.lock # The lock file for the dependencies
|
```

## Usage

To work with the project you need to install:

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Afterwards run use uv to install [poethepoet](`https://poethepoet.natn.io`):

```
uv tool install poethepoet
```

Then you can install the project dependencies using:

```
uv sync
```

And create a new experiment using:

```
poe experiment experiment_name
```

## Get the Dataset

The dataset is not included in this repository and must be downloaded separately (e.g., from the FAUDataCloud).

The dataset can be placed anywhere on your system, you just need to specify the path in a config file.

To do so, create a new file called `config.json` in the `experiments` folder of this repository. The file should follow
the following structure:

```json
{
    "<deploy_type_1>": {
        "base_path": "path-to-dataset"
    },
    "deploy_type_2": {
        "base_path": ""
    }
}
```

Hereby, the `deploy_type` can be used to specify different locations of datasets, e.g., `local`, 
`external_drive`, or `remote`.

Afterwards, the path to the dataset can be loaded in the notebooks by specifying the `deploy_type`.


## Development Information

### Dependency management

All dependencies are manged using `uv`.
uv will automatically create a new venv for the project, when you run `uv sync`.
Check out the [documentation](https://docs.astral.sh/uv/) on how to add and remove dependencies.


### Jupyter Notebooks

To use jupyter notebooks with the project you need to add a jupyter kernel pointing to the venv of the project.
This can be done by running:

```
poe conf_jupyter
```

Afterwards a new kernel called `stressgait_analysis` should be available in the jupyter lab / jupyter notebook interface.
Use that kernel for all notebooks related to this project.



All jupyter notebooks should go into the `notebooks` subfolder of the respective experiment.
To make best use of the folder structure, the parent folder of each notebook should be added to the import path.
This can be done by adding the following lines to your first notebook cell:

```python
# Optional: Auto reloads the helper and the main stressgait_analysis module
%load_ext
autoreload
%autoreload
2

from stressgait_analysis import conf_rel_path

conf_rel_path()
```

This allows to then import the helper and the script module belonging to a specific experiment as follows:

```
import helper
# or
from helper import ...
```

### Format and Linting

To ensure consistent code structure this project uses prospector, black, and ruff to automatically check (and fix) the code format.

```
poe format  # runs ruff format and ruff lint with the autofix flag
poe lint # runs ruff without autofix (will show issues that can not automatically be fixed)
```

If you want to check if all code follows the code guidelines, run `poe ci_check`.
This can be useful in the CI context


### Tests

All tests are located in the `tests` folder and can be executed by using `poe test`.
