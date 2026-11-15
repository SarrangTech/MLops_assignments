# data/

`src/data_pipeline.py` writes three generated files here at run
time, all from scikit-learn's California housing dataset (20,640
census block groups; 8 features such as median income, house age,
and average rooms; target = median house value in $100k):

- `train.csv` - the training split (13,209 rows, 64%)
- `validation.csv` - the validation split (3,303 rows, 16%)
- `test.csv` - the held-out test split (4,128 rows, 20%)

Both splits use seed 42. The test split is drawn first, exactly as
the earlier two-way version of this pipeline drew it, so the test
rows are unchanged from that version; the validation split is
carved out of the remaining 80%.

The three-way split keeps the reported champion score honest: the
experiment runs are compared, and the promotion gate is decided, on
validation metrics (`val_rmse`), and the test split is touched only
afterwards - once, to score the finished champion - plus by the
serving check, which compares predictions on its first 20 rows
without selecting anything on them.

The CSVs are git-ignored on purpose: the splits are reproducible
from the seed, and every stage of the project reads these exact
files, so the tracked metrics, the promotion gate, and the champion
scoring all refer to the same rows.
