import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as scs

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve
from sklearn.preprocessing import PolynomialFeatures, Imputer, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report

sets = []

def read_files(files):
    for file in files:
        sets.append(pd.read_csv(file))

def feature_engineering():

    # Borrower Characteristics
    for i in sets:
        i["AGE"] = sets[0]["DAYS_BIRTH"]/-365
        i["AGESQ"] = sets[0]["AGE"]**2

    for i in sets:
        i_dummies=pd.get_dummies(i[['CODE_GENDER', 'NAME_EDUCATION_TYPE',
                                'NAME_FAMILY_STATUS', 'NAME_TYPE_SUITE', 
                                    'NAME_INCOME_TYPE']],drop_first=True)
        i[i_dummies.columns]=i_dummies


    # Employment 
    for i in sets:
        i_dummies=pd.get_dummies(i[['OCCUPATION_TYPE']],drop_first=True)
        i[i_dummies.columns]=i_dummies

    for i in sets:
        i['employed'] = i['DAYS_EMPLOYED']!=365243 *1
        i['YEARS_EMPLOYED'] = i['DAYS_EMPLOYED']/-365
        i["INCOME_LOG"] = np.log(i["AMT_INCOME_TOTAL"])
        i['employed_years']=i['employed']* i['YEARS_EMPLOYED']


    # Assets 
    for i in sets:
        i_dummies=pd.get_dummies(i[['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                                'NAME_HOUSING_TYPE']],drop_first=True)
        i[i_dummies.columns]=i_dummies

    for i in sets:
        i['FLAG_OWN_CAR_CAR_AGE']=i['FLAG_OWN_CAR_Y']*i['OWN_CAR_AGE']

    
    # Credit 
    for i in sets:
        i_dummies=pd.get_dummies(i[['NAME_CONTRACT_TYPE']],drop_first=True)
        i[i_dummies.columns]=i_dummies

    for i in sets:
        i["AMT_CREDIT_LOG"] = np.log(i["AMT_CREDIT"]) # this is the amount of the loan
        i["AMT_GOODS_PRICE_LOG"] =np.log(i['AMT_GOODS_PRICE']) # FOR CONS LOANS-- half million usd???
        i["AMT_ANNUITY_LOG"] =np.log(i['AMT_ANNUITY'])
        i['CREDIT_INCOME'] = i["AMT_CREDIT"]/i["AMT_INCOME_TOTAL"]
        i['ANNUITY_INCOME'] = i['AMT_ANNUITY']/i["AMT_INCOME_TOTAL"]

    docs =  ['FLAG_DOCUMENT_2',  'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

    for i in sets:
        i['DOCS'] = 0
        for d in docs:
            sets[0]['DOCS'] += sets[0][d]
    
def modeling():

    # Train
    df_train = sets[0][['TARGET','AGE', 'AGESQ','REGION_RATING_CLIENT_W_CITY',
                        'DAYS_EMPLOYED','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3',
                        'CODE_GENDER_M', 'NAME_EDUCATION_TYPE_Higher education',
                        'NAME_EDUCATION_TYPE_Incomplete higher',
                        'NAME_EDUCATION_TYPE_Lower secondary',
                        'NAME_EDUCATION_TYPE_Secondary / secondary special',
                        'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated',
                        'NAME_FAMILY_STATUS_Single / not married', 
                        'NAME_FAMILY_STATUS_Widow', 'NAME_TYPE_SUITE_Family',
                        'NAME_TYPE_SUITE_Group of people', 'NAME_TYPE_SUITE_Other_A',
                        'NAME_TYPE_SUITE_Other_B', 'NAME_TYPE_SUITE_Spouse, partner',
                        'NAME_TYPE_SUITE_Unaccompanied',
                        'NAME_INCOME_TYPE_Commercial associate',
                        'NAME_INCOME_TYPE_Pensioner',
                        'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Student',
                        'NAME_INCOME_TYPE_Unemployed', 'NAME_INCOME_TYPE_Working',
                        'OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff',
                        'OCCUPATION_TYPE_Core staff', 'OCCUPATION_TYPE_Drivers',
                        'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_High skill tech staff',
                        'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Laborers',
                        'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Managers',
                        'OCCUPATION_TYPE_Medicine staff',
                        'OCCUPATION_TYPE_Private service staff',
                        'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff',
                        'OCCUPATION_TYPE_Secretaries', 'OCCUPATION_TYPE_Security staff',
                        'OCCUPATION_TYPE_Waiters/barmen staff',
                        'employed', 'employed_years',
                        'INCOME_LOG',
                        'FLAG_OWN_CAR_Y', 'FLAG_OWN_REALTY_Y',
                        'NAME_HOUSING_TYPE_House / apartment',
                        'NAME_HOUSING_TYPE_Municipal apartment',
                        'NAME_HOUSING_TYPE_Office apartment',
                        'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_With parents',
                        'FLAG_OWN_CAR_CAR_AGE', 'NAME_CONTRACT_TYPE_Revolving loans',
                        'AMT_CREDIT_LOG', 'AMT_GOODS_PRICE_LOG', 'AMT_ANNUITY_LOG',
                        'CREDIT_INCOME', 'ANNUITY_INCOME', 'DOCS']]

    X_train = df_train.drop('TARGET',axis=1).values
    y_train = df_train['TARGET'].values

    # Test
    X_test = sets[1][['AGE', 'AGESQ','REGION_RATING_CLIENT_W_CITY',
                    'DAYS_EMPLOYED','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3',
                    'CODE_GENDER_M', 'NAME_EDUCATION_TYPE_Higher education',
                    'NAME_EDUCATION_TYPE_Incomplete higher',
                    'NAME_EDUCATION_TYPE_Lower secondary',
                    'NAME_EDUCATION_TYPE_Secondary / secondary special',
                    'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated',
                    'NAME_FAMILY_STATUS_Single / not married',
                    'NAME_FAMILY_STATUS_Widow', 'NAME_TYPE_SUITE_Family',
                    'NAME_TYPE_SUITE_Group of people', 'NAME_TYPE_SUITE_Other_A',
                    'NAME_TYPE_SUITE_Other_B', 'NAME_TYPE_SUITE_Spouse, partner',
                    'NAME_TYPE_SUITE_Unaccompanied',
                    'NAME_INCOME_TYPE_Commercial associate',
                    'NAME_INCOME_TYPE_Pensioner',
                    'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Student',
                    'NAME_INCOME_TYPE_Unemployed', 'NAME_INCOME_TYPE_Working',
                    'OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff',
                    'OCCUPATION_TYPE_Core staff', 'OCCUPATION_TYPE_Drivers',
                    'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_High skill tech staff',
                    'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Laborers',
                    'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Managers',
                    'OCCUPATION_TYPE_Medicine staff',
                    'OCCUPATION_TYPE_Private service staff',
                    'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff',
                    'OCCUPATION_TYPE_Secretaries', 'OCCUPATION_TYPE_Security staff',
                    'OCCUPATION_TYPE_Waiters/barmen staff',
                    'employed', 'employed_years',
                    'INCOME_LOG',
                    'FLAG_OWN_CAR_Y', 'FLAG_OWN_REALTY_Y',
                    'NAME_HOUSING_TYPE_House / apartment',
                    'NAME_HOUSING_TYPE_Municipal apartment',
                    'NAME_HOUSING_TYPE_Office apartment',
                    'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_With parents',
                    'FLAG_OWN_CAR_CAR_AGE', 'NAME_CONTRACT_TYPE_Revolving loans',
                    'AMT_CREDIT_LOG', 'AMT_GOODS_PRICE_LOG', 'AMT_ANNUITY_LOG',
                    'CREDIT_INCOME', 'ANNUITY_INCOME', 'DOCS']]

    X_test = X_test.values 

    # Logistic 
    imp = Imputer(strategy='median') 
    imp.fit(X_train) 

    # transform the test & train data
    X_train=imp.transform(X_train)
    X_test=imp.transform(X_test)


    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print('Training data shape: ', X_train.shape)
    print('Testing data shape: ', X_test.shape)

    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train,y_train)
    # p = model.predict_proba(X_train)[:,1]
    probabilities = model.predict_proba(X_test)[:,1]

    return probabilities

def submission(probabilities):
    submit = sets[1][['SK_ID_CURR']]
    submit['TARGET'] = probabilities
    # submit.head()
    submit.to_csv('logistic_mar_solutions.csv', index = False)


if __name__ == '__main__':
    print("$ READ FILES")
    read_files(['application_train.csv.zip', 'application_test.csv.zip'])

    print("$ FEATURE ENGINEERING")
    feature_engineering()

    print("$ MODELING")
    prob = modeling()

    print("$ SUBMISSION")
    submission(prob)
