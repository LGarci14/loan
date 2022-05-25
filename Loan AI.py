#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import plot_precision_recall_curve
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report
from lightgbm import LGBMClassifier


# In[2]:


df = pd.read_csv('credit_risk_dataset.csv')
df.head()


# In[3]:


dups = df.duplicated()


# In[4]:


df[dups]


# In[5]:


df.query("person_age==23 & person_income==42000 &person_home_ownership=='RENT' & loan_int_rate==9.99")


# In[6]:


df.shape


# In[7]:


df.drop_duplicates(inplace=True)


# In[8]:


df.shape


# In[9]:


# X and y will be thought of as the entire training data
# X_test and y_test will be thought of as the out of sample data for model evaluation

X, X_test, y, y_test = train_test_split(df.drop('loan_status', axis=1), df['loan_status'],
                                        random_state=0,  test_size=0.2, stratify=df['loan_status'],
                                        shuffle=True)


# In[10]:


df['loan_status'].value_counts(normalize=True)


# In[11]:


y.value_counts(normalize=True)


# In[12]:


y_test.value_counts(normalize=True)


# In[13]:


np.round(X.isna().sum()* 100 / X.shape[0], 3)


# In[14]:


X.shape


# In[15]:


X.dropna().shape


# In[16]:


(25932-22763)/25932


# In[17]:



X[['person_income', 'loan_amnt', 'loan_percent_income']].head()


# In[18]:


X.drop('loan_percent_income', axis=1, inplace=True)
X_test.drop('loan_percent_income', axis=1, inplace=True)


# In[19]:


for col in X:
    print(col, '--->', X[col].nunique())
    if X[col].nunique()<20:
        print(X[col].value_counts(normalize=True)*100)
    print()


# In[20]:


X.describe()


# In[21]:


num_cols = [col for col in X if X[col].dtypes != 'O']
num_cols


# In[ ]:


for col in num_cols:
    sns.histplot(X[col])
    plt.show()


# In[23]:


X.loc[X['person_age']>=80, :]


# In[24]:


X = X.loc[X['person_age']<80, :]


# In[25]:


X.shape


# In[26]:


X.loc[X['person_emp_length']>=66, :]


# In[27]:


df.query("person_age<=person_emp_length+14")


# In[28]:


X = X.loc[(X['person_emp_length']<66) | (X['person_emp_length'].isna()), :]


# In[29]:


# since we've removed some data from X, we need to pass on these updations to y as well,
# as y doesn't know some of its corresponding X's have been deleted.
y = y[X.index]


# In[30]:


cat_cols = [col for col in X if X[col].dtypes == 'O']
cat_cols


# In[31]:


num_pipe = Pipeline([
    ('impute', IterativeImputer()),
    ('scale', StandardScaler()),
])


# In[32]:


ct = ColumnTransformer([
    ('num_pipe', num_pipe, num_cols),
    ('cat_cols', OneHotEncoder(sparse=False, handle_unknown='ignore'), cat_cols)
], remainder='passthrough')


# In[ ]:


grid = {
    RandomForestClassifier(random_state=0, n_jobs=-1, class_weight='balanced'):
    {'model__n_estimators':[300,400,500],
     'coltf__num_pipe__impute__estimator': [LinearRegression(), RandomForestRegressor(random_state=0),
                                        KNeighborsRegressor()]},
    
    LGBMClassifier(class_weight='balanced', random_state=0, n_jobs=-1):
    {'model__n_estimators':[300,400,500],
     'model__learning_rate':[0.001,0.01,0.1,1,10],
     'model__boosting_type': ['gbdt', 'goss', 'dart'],
     'coltf__num_pipe__impute__estimator':[LinearRegression(), RandomForestRegressor(random_state=0),
                                        KNeighborsRegressor()]},
}


# In[ ]:


for clf, param in grid.items():
    print(clf)
    print('-'*50)
    print(param)
    print('\n')


# In[ ]:


full_df = pd.DataFrame()
best_algos = {}

for clf, param in grid.items():
    pipe = Pipeline([
    ('coltf', ct),
    ('model', clf)
])

    gs = RandomizedSearchCV(estimator=pipe, param_distributions=param, scoring='accuracy',
                            n_jobs=-1, verbose=3, n_iter=4, random_state=0)
    
    gs.fit(X, y)
    
    all_res = pd.DataFrame(gs.cv_results_)

    temp = all_res.loc[:, ['params', 'mean_test_score']]
    algo_name = str(clf).split('(')[0]
    temp['algo'] = algo_name
    
    full_df = pd.concat([full_df, temp], ignore_index=True)
    best_algos[algo_name] = gs.best_estimator_


# In[ ]:


full_df.sort_values('mean_test_score', ascending=False)


# In[ ]:


full_df.sort_values('mean_test_score', ascending=False).iloc[0, 0]


# In[ ]:


be = best_algos['RandomForestClassifier']
be


# In[ ]:


be.fit(X, y)


# In[ ]:


preds = be.predict(X_test)


# In[ ]:


confusion_matrix(y_test, preds)


# In[ ]:


plot_confusion_matrix(be, X_test, y_test)


# In[ ]:


print(classification_report(y_test, preds))


# In[ ]:


be.score(X_test, y_test)


# In[ ]:


plot_precision_recall_curve(estimator=be, X=X_test, y=y_test, name='model AUC')
baseline = y_test.sum() / len(y_test)
plt.axhline(baseline, ls='--', color='r', label=f'Baseline model ({round(baseline,2)})')
plt.legend(loc='best')


# In[ ]:


a, b, c = learning_curve(be, X, y, n_jobs=-1, scoring='accuracy')


# In[ ]:


plt.plot(a, b.mean(axis=1), label='training accuracy')
plt.plot(a, c.mean(axis=1),  label='validation accuracy')
plt.xlabel('training sample sizes')
plt.ylabel('accuracy')
plt.legend()


# In[ ]:


grid = {
    
    RandomForestClassifier(random_state=0, n_jobs=-1, class_weight='balanced'):
    {'model__n_estimators':[100,200,300],
     'model__max_depth':[5, 9, 13],
     'model__min_samples_split':[4,6,8],
     'coltf__num_pipe__impute__estimator': [LinearRegression(), RandomForestRegressor(random_state=0),
                                        KNeighborsRegressor()]},
    
#     LGBMClassifier(class_weight='balanced', random_state=0, n_jobs=-1):
#     {'model__n_estimators':[100,200,300],
#      'model__max_depth':[5, 9, 13],
#      'model__num_leaves': [7,15,31],
#      'model__learning_rate':[0.0001,0.001,0.01,0.1,],
#      'model__boosting_type': ['gbdt', 'goss', 'dart'],
#      'coltf__num_pipe__impute__estimator':[LinearRegression(), RandomForestRegressor(random_state=0),
#                                         KNeighborsRegressor()]} 
}


# In[ ]:


for clf, param in grid.items():
    print(clf)
    print('-'*50)
    print(param)
    print('\n')


# In[ ]:


full_df = pd.DataFrame()
best_algos = {}

for clf, param in grid.items():
    pipe = Pipeline([
    ('coltf', ct),
    ('model', clf)
])

    gs = RandomizedSearchCV(estimator=pipe, param_distributions=param, scoring='accuracy',
                            n_jobs=-1, verbose=3, n_iter=4)
    
    gs.fit(X, y)
    
    all_res = pd.DataFrame(gs.cv_results_)

    temp = all_res.loc[:, ['params', 'mean_test_score']]
    algo_name = str(clf).split('(')[0]
    temp['algo'] = algo_name
    
    full_df = pd.concat([full_df, temp])
    best_algos[algo_name] = gs.best_estimator_


# In[ ]:


full_df.sort_values('mean_test_score', ascending=False)


# In[ ]:


be = best_algos['RandomForestClassifier']
be


# In[ ]:


be.fit(X, y)


# In[ ]:


preds = be.predict(X_test)


# In[ ]:


confusion_matrix(y_test, preds)


# In[ ]:


plot_confusion_matrix(be, X_test, y_test)


# In[ ]:


print(classification_report(y_test, preds))


# In[ ]:


be.score(X_test, y_test)


# In[ ]:


plot_precision_recall_curve(be, X_test, y_test)
baseline = y_test.sum() / len(y_test)
plt.axhline(baseline, ls='--', color='r', label=f'Baseline model ({round(baseline,2)})')
plt.legend(loc='best')


# In[ ]:


a, b, c = learning_curve(be, X, y, n_jobs=-1, cv=5)


# In[ ]:


a


# In[ ]:


b


# In[ ]:


c


# In[ ]:


plt.plot(a, b.mean(axis=1), label='training accuracy')
plt.plot(a, c.mean(axis=1),  label='validation accuracy')
plt.xlabel('training sample sizes')
plt.ylabel('accuracy')
plt.legend()

