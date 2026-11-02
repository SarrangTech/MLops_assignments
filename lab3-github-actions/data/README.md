# data/

`src/data_pipeline.py` writes two generated files here at run time:

- `train.csv` — the training split (353 rows)
- `test.csv` — the held-out test split (89 rows)

Both come from scikit-learn's diabetes dataset (442 patients, 10
baseline features, target = disease progression after one year),
split 80/20 with seed 42 and stratified by target quintile. They are
git-ignored on purpose: the split is reproducible from the seed, and
CI regenerates them on every run so the evaluated rows are always
the ones the current code produced.
