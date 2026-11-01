# data/

In the course lab's folder structure, `data/` holds the project's data
files. This submission's datasets need no files here:

- the calculator project (the lab itself) uses no data;
- the training pipeline's digits dataset ships with scikit-learn, so
  nothing is downloaded or stored.

The folder is kept so the layout matches the lab's structure
(`data/`, `src/`, `test/`). It deliberately has no `__init__.py`:
the training pipeline's module is `../data.py`, and a `data` package
here would shadow it on import.
