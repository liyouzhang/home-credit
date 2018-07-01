import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as scs

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve
from sklearn.preprocessing import PolynomialFeatures, Imputer, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, PredefinedSplit
from sklearn.metrics import classification_report, make_scorer

from sklearn.pipeline import Pipeline
from pipeline_part1 import Getdummies, Logarize, DatetoYear, SelectColumns, ReplaceNaN, Square, Interactify

sets = []

def read_files(files):
    for file in files:
        sets.append(pd.read_csv(file))

def rmsle(y_true, y_pred):
    """Calculate the root mean squared log error between y
    predictions and true ys.
    (hard-coding the clipping here as a dumb hack for the Pipeline)
    """
    y_pred_clipped = np.clip(y_pred, 4750, None)
    log_diff = np.log(y_true+1) - np.log(y_pred_clipped+1) 
    return np.sqrt(np.mean(log_diff**2))

def modeling():

    X = sets[0].drop('TARGET',axis=1)
    y = sets[0]['TARGET']

    # crossval = PredefinedSplit()
    # X_test = sets[1]

    p = Pipeline([
        ('fillna',ReplaceNaN()),
        ('getdummies',Getdummies()),
        ('log',Logarize()),
        ('square', Square()),
        ('datetoyear',DatetoYear()),
        ('interactions', Interactify()),
        ('select',SelectColumns()),
        ('lg',LogisticRegression())
        ])

    # X = X.reset_index()

    # p.fit(X,y)
    # p.transform(X)

    # X_test = X_test.values 

    # # Logistic 
    # imp = Imputer(strategy='median') 
    # imp.fit(X) 

    # # transform the test & train data
    # X_=imp.transform(X)
    # X_test=imp.transform(X_test)


    # scaler = MinMaxScaler(feature_range = (0, 1))
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    # print('Training data shape: ', X_train.shape)
    # print('Testing data shape: ', X_test.shape)
    params = {'lg__class_weight':['balanced',None]}

    rmsle_scorer = make_scorer(rmsle,greater_is_better=False)
    gscv = GridSearchCV(p,param_grid=params,scoring=rmsle_scorer,cv=3,n_jobs=-1)
    clf = gscv.fit(X.reset_index(),y)
    
    print('Best parameters: {}'.format(clf.best_params_))
    print('Best RMSLE: {}'.format(clf.best_score_))

    test = sets[1]
    # X_test = test.sort_values(by='SalesID')

    probabilites = clf.predict(test)
    # probabilites = test_predictions
    # outfile = 'solution_benchmark.csv'
    # test[['SalesID', 'SalePrice']].to_csv(outfile, index=False)
    
    # model = LogisticRegression(class_weight='balanced')
    # model.fit(X_train,y_train)
    # p = model.predict_proba(X_train)[:,1]
    # probabilities = model.predict_proba(X_test)[:,1]

    return probabilites

def submission(probabilities):
    submit = sets[1][['SK_ID_CURR']]
    submit['TARGET'] = probabilities
    # submit.head()
    submit.to_csv('logistic_mar_solutions.csv', index = False)


if __name__ == '__main__':
    print("$ READ FILES")
    read_files(['application_train.csv.zip', 'application_test.csv.zip'])

    # print("$ FEATURE ENGINEERING")
    # feature_engineering()

    print("$ MODELING")
    prob = modeling()

    print("$ SUBMISSION")
    submission(prob)
