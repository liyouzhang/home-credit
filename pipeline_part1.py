from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

class Getdummies(BaseEstimator, TransformerMixin):

    # col_names = ['CODE_GENDER', 'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS', 'NAME_TYPE_SUITE','NAME_INCOME_TYPE','OCCUPATION_TYPE','FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    #                        'NAME_HOUSING_TYPE','NAME_CONTRACT_TYPE']
    selected = {
        'CODE_GENDER':['CODE_GENDER_M']
        'NAME_EDUCATION_TYPE':,
        'NAME_FAMILY_STATUS':,
        'NAME_FAMILY_STATUS':,
        'NAME_TYPE_SUITE':,
        'NAME_INCOME_TYPE':
    }
    def fit(self,X,y):
    #X is a dataframe
        return self

    def transform(self,X):
        for col in self.selected.keys():
            dummies = pd.get_dummies(X[col],prefix=col)
            selected_cols = self.selected[col] #['Male']
            selected_dummies = dummies[selected_cols]
            X[selected_dummies.columns] = selected_dummies
        return X
