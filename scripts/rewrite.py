import pandas as pd
tpl = pd.read_csv('../data/crossval_splits/all_amine_split_for_paper/cv_0/train_extra_x.csv', nrows=0)
cols = list(tpl.columns)
df = pd.read_csv('../data/libraries/Chinese_Academy_of_Sciences/Chinese_Academy_of_Sciences_extra_x.csv')

df = df.reindex(columns=cols, fill_value=0.0)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
df.to_csv('../data/libraries/Chinese_Academy_of_Sciences/Chinese_Academy_of_Sciences_extra_x.csv', index=False)