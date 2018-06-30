# %load_ext autoreload
# %autoreload 2
import pandas as pd
from pipeline_part1 import Getdummies
X = pd.read_csv('application_train.csv.zip')
y = X['TARGET']
g = Getdummies()
model = g.fit(X,y)
model.transform(X)