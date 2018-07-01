# import matplotlib.pyplot as plt
# import seaborn as sns
# import statsmodels.api as sm
# import scipy.stats as scs

# %matplotlib inline
# plt.style.use('ggplot') # overall 'ggplot' style

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

# %load_ext autoreload
# %autoreload 2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline


import pandas as pd
from pipeline_part1 import Getdummies, Logarize, DatetoYear, SelectColumns, ReplaceNaN, Square, Interactify
X = pd.read_csv('application_train.csv.zip')
y = X['TARGET']

p = Pipeline([
    ('select',SelectColumns()),
    ('fillna',ReplaceNaN()),
    ('getdummies',Getdummies()),
    ('log',Logarize()),
    ('square', Square()),
    ('datetoyear',DatetoYear()),
    ('interactions', Interactify())
])

X = X.reset_index()

p.fit(X,y)
p.transform(X)

# g = Getdummies()
# model = g.fit(X,y)
# model.transform(X)

# l = Logarize()
# model = l.fit(X,y)
# model.transform(X)


# l = Logarize()
# model = l.fit(X,y)
# model.transform(X)

print(X.columns)