# Development

To create a development environment, you must have [`mamba` installed](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

A development conda environment can be created and activated with:

```shell
make env
conda activate femto
```

To format the codebase:

```shell
make format
```

To run the unit tests:

```shell
make test
```

To serve the documentation locally:

```shell
mkdocs serve
```
