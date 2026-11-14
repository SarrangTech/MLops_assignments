# data/

`src/data_pipeline.py` writes two generated files here at run time:

- `train.csv` — the training split (16,512 rows)
- `test.csv` — the held-out test split (4,128 rows)

Both come from scikit-learn's California housing dataset (20,640
census block groups; 8 features such as median income, house age,
and average rooms; target = median house value in $100k), split
80/20 with seed 42. They are git-ignored on purpose: the split is
reproducible from the seed, and every stage of the project reads
these exact files, so the tracked metrics, the promotion gate, and
the champion scoring all refer to the same rows.
