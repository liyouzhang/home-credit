from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

family=['NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated',
      'NAME_FAMILY_STATUS_Single / not married']

education = ['NAME_EDUCATION_TYPE_Higher education',
      'NAME_EDUCATION_TYPE_Incomplete higher',
      'NAME_EDUCATION_TYPE_Lower secondary',
      'NAME_EDUCATION_TYPE_Secondary / secondary special']

housing = ['NAME_HOUSING_TYPE_Municipal apartment',
      'NAME_HOUSING_TYPE_Office apartment',
      'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_With parents']

contract = ['NAME_CONTRACT_TYPE_Revolving loans']

gender = ['CODE_GENDER_M']

# selected = ['CODE_GENDER','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_CONTRACT_TYPE': contract,
#         'NAME_HOUSING_TYPE':housing
#         # 'OCCUPATION_TYPE':[],
#         # 'FLAG_OWN_CAR':[],
#         # 'FLAG_OWN_REALTY':[]
#         }.keys()


selected_for_dummies = {
    'CODE_GENDER': gender,
    'NAME_EDUCATION_TYPE':education,
    'NAME_FAMILY_STATUS':family,
    # 'NAME_TYPE_SUITE':[],
    # 'NAME_INCOME_TYPE':[],
    'NAME_CONTRACT_TYPE': contract,
    'NAME_HOUSING_TYPE':housing,
    # 'OCCUPATION_TYPE':[],
    'FLAG_OWN_CAR':['FLAG_OWN_CAR_Y'],
    'FLAG_OWN_REALTY':['FLAG_OWN_REALTY_Y']
    }

class SelectColumns(BaseEstimator, TransformerMixin):
    """Only keep columns that we want to keep.
    """
    temp = ['TARGET','AMT_ANNUITY','OWN_CAR_AGE','NAME_HOUSING_TYPE','DAYS_EMPLOYED', 'DAYS_BIRTH', 'DAYS_REGISTRATION','AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY','AGE']
    keep_cols = temp + family + education + housing + gender + contract
    keep_cols = temp + list(selected_for_dummies.keys())
    keep_cols = list(set(keep_cols))
    # drop_cols = []

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.loc[:, self.keep_cols]


class Getdummies(BaseEstimator, TransformerMixin):

    # col_names = ['CODE_GENDER', 'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS', 'NAME_TYPE_SUITE','NAME_INCOME_TYPE','OCCUPATION_TYPE','FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    #                        'NAME_HOUSING_TYPE','NAME_CONTRACT_TYPE']

    #selected based correlation analysis (0.02)

    # selected = {
    #     'CODE_GENDER': gender,
    #     'NAME_EDUCATION_TYPE':education,
    #     'NAME_FAMILY_STATUS':family,
    #     # 'NAME_TYPE_SUITE':[],
    #     # 'NAME_INCOME_TYPE':[],
    #     'NAME_CONTRACT_TYPE': contract,
    #     'NAME_HOUSING_TYPE':housing
    #     # 'OCCUPATION_TYPE':[],
    #     # 'FLAG_OWN_CAR':[],
    #     # 'FLAG_OWN_REALTY':[]
    #     }

    def fit(self,X,y):
    #X is a dataframe
        return self

    def transform(self,X):
        for col in selected_for_dummies.keys():
            dummies = pd.get_dummies(X[col],prefix=col)
            selected_cols = selected_for_dummies[col] #['Male']
            selected_dummies = dummies[selected_cols]
            X[selected_dummies.columns] = selected_dummies
        return X


class Logarize(BaseEstimator, TransformerMixin):

    columns_to_log = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY']

    def fit(self,X,y):
        return self

    def transform(self,X):
        for col in self.columns_to_log:
            X[col + '_log'] = np.log(X[col])
        return X


class Square(BaseEstimator, TransformerMixin):

    columns_to_square = ['AGE']

    def fit(self,X,y):
        return self

    def transform(self,X):
        for col in self.columns_to_square:
            X[col + '_square'] = X[col]**2
        return X


class DatetoYear(BaseEstimator, TransformerMixin):

    columns_to_convert = ['DAYS_EMPLOYED', 'DAYS_BIRTH', 'DAYS_REGISTRATION']

    def fit(self,X,y):
        return self

    def transform(self,X):
        for col in self.columns_to_convert:
            X[col + '_year'] = X[col] / (-365)
        return X


# class DataType(BaseEstimator, TransformerMixin):
#     col_types = {}
#     def fit(self,X,y):
#         return self
#     def transform(self,X):
#         for col_type, column in self.col_types.items():
#             X[column] = X[column].astype(col_type)
#         X    



class ReplaceNaN(BaseEstimator, TransformerMixin):
    """Replace NaNs
    """
    num_col_name = ['OWN_CAR_AGE']
    cat_col_name = ['NAME_HOUSING_TYPE']
    

    def fit(self, X, y):
        ''' which columns are numerical and which columns are categorical'''
        
        self.dict = {}
        num_median = X[self.num_col_name].median().values.flatten()
        cat_mod = X[self.cat_col_name].mode().values.flatten()
        for col_name, value in zip(self.cat_col_name, cat_mod):
            self.dict[col_name] = value
        for col_name, value in zip(self.num_col_name, num_median):
            self.dict[col_name] = value

        print(self.dict)
        return self

    def transform(self, X):
        # print(X.columns)
        X.fillna(value=self.dict, inplace=True)
        # print(X.columns)
        return X

class Interactify(BaseEstimator, TransformerMixin):
    ''' Interactions '''

    # interactifier1 = ['FLAG_OWN_CAR_Y']
    # interactifier2 = ['OWN_CAR_AGE']

    def __init__(self, list1, list2):
        self.interactifier1 = list1
        self.interactifier2 = list2
        super() 

    def fit(self, X, y):
        return self

    def transform(self, X):
        for i, j in zip(self.interactifier1, self.interactifier2):
            X[i+"_"+j] = X[i] * X[j]

        return X