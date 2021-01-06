#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[2]:


HRSS_A_OPT = pd.read_csv("C:\\Users\\vedan\\Desktop\\ML ASSIGNMENT\\HRSS\\HRSS_anomalous_optimized.csv")
HRSS_N_OPT = pd.read_csv("C:\\Users\\vedan\\Desktop\\ML ASSIGNMENT\\HRSS\\HRSS_normal_optimized.csv")


# In[3]:


HRSS_A_STA = pd.read_csv("C:\\Users\\vedan\\Desktop\\ML ASSIGNMENT\\HRSS\\HRSS_anomalous_standard.csv")
HRSS_N_STA = pd.read_csv("C:\\Users\\vedan\\Desktop\\ML ASSIGNMENT\\HRSS\\HRSS_normal_standard.csv")


# **Missing Values**

# In[4]:


print(HRSS_A_OPT.isna().sum())
print(HRSS_N_OPT.isna().sum())


# **Calculating the total cycles**

# In[5]:


def cycletime(df):
    df["CYCLE"]=np.nan
    df["TIME"]=np.nan
    df["TPOWER"]=df["O_w_BLO_power"]+df["O_w_BHL_power"]+df["O_w_BHR_power"]+df["O_w_BRU_power"]+df["O_w_HR_power"]+df["O_w_HL_power"]
    grp=0
    time=0
    for i in range(len(df)):
        l=i-1
        if(df["Timestamp"][i]==0):
            if(grp!=0):
                df["TIME"][i]=df["TIME"][l]+df["Timestamp"][i]
            else:
                df["TIME"][i]=df["Timestamp"][i]
            grp=grp+1
        else:
            df["TIME"][i]=df["TIME"][l]+df["Timestamp"][i]
        df['CYCLE'][i]=grp
    return df


# In[6]:


HRSS_A_OPT=cycletime(HRSS_A_OPT)
HRSS_N_OPT=cycletime(HRSS_N_OPT)


# In[7]:


HRSS_A_OPT


# In[8]:


HRSS_A_STA=cycletime(HRSS_A_STA)
HRSS_N_STA=cycletime(HRSS_N_STA)


# In[9]:



plt.figure(figsize=(10, 6))
sns.scatterplot(HRSS_A_OPT["TIME"],HRSS_A_OPT["Labels"],label="OPTIMISED")
sns.scatterplot(HRSS_A_STA["TIME"],HRSS_A_STA["Labels"]+0.5,label="STANDARD")
#sns.lineplot(range(1,108),TPOWER_STA,label="STANDARD")
plt.legend()
plt.title("ANOMALY")
plt.plot()


# In[ ]:


HRSS_A_OPT_SUM=HRSS_A_OPT.groupby('CYCLE').sum()
HRSS_N_OPT_SUM=HRSS_N_OPT.groupby('CYCLE').sum()
TPOWER_OPT=HRSS_A_OPT_SUM['TPOWER']+HRSS_N_OPT_SUM['TPOWER']


# In[ ]:


HRSS_A_STA_SUM=HRSS_A_STA.groupby('CYCLE').sum()
HRSS_N_STA_SUM=HRSS_N_STA.groupby('CYCLE').sum()
TPOWER_STA=HRSS_A_STA_SUM['TPOWER']+HRSS_N_STA_SUM['TPOWER']


# In[ ]:


plt.figure(figsize=(10, 6))
sns.lineplot(range(1,112),TPOWER_OPT,label="OPTIMISED")
sns.lineplot(range(1,108),TPOWER_STA,label="STANDARD")
plt.legend()
plt.title("POWER CONSUMPTION")
plt.plot()


# In[ ]:





# In[11]:


corr=HRSS_A_OPT.drop(['Timestamp','Labels','TPOWER','TIME'],axis='columns').corr()
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr,cmap='coolwarm',square=True, linewidths=.5)


# **Dropping unwanted columns**

# In[12]:


HRSS_A_OPT_inputs = HRSS_A_OPT.drop(['Labels','O_w_BLO_voltage','O_w_BHL_voltage','O_w_BHR_voltage','O_w_BRU_voltage','O_w_HR_voltage','O_w_HL_voltage','CYCLE','TPOWER','TIME'],axis='columns')
HRSS_A_OPT_targets=HRSS_A_OPT['Labels']
HRSS_N_OPT_inputs = HRSS_N_OPT.drop(['Labels','O_w_BLO_voltage','O_w_BHL_voltage','O_w_BHR_voltage','O_w_BRU_voltage','O_w_HR_voltage','O_w_HL_voltage','CYCLE','TPOWER','TIME'],axis='columns')
HRSS_N_OPT_targets=HRSS_N_OPT['Labels']
X_train,X_test,y_train,y_test=train_test_split(HRSS_A_OPT_inputs, HRSS_A_OPT_targets, test_size = 0.2, random_state = 3)


# In[13]:


def model_scores(model):
    model.fit(X_train,y_train)
    model.score(X_train,y_train)
    print("Model Accuracy:",model.score(X_train,y_train))
    Predictions = model.predict(X_test)
    #print(Predictions)
    print(classification_report(y_test,Predictions))
    print(confusion_matrix(y_test, Predictions))
    cm=confusion_matrix(y_test, Predictions)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='salmon')
    plt.title("Confusion Matrix")
    plt.show()


# In[14]:


def model_score_grid(model):
    model.fit(X_train,y_train)
    print(model.best_params_)


# <h2 style='color:salmon' align="center">Logistic Regression</h2>

# In[15]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
model_scores(log_model)


# <h2 style='color:blue' align="center">Decision Tree Classification</h2>

# In[16]:


from sklearn import tree
DT_model = tree.DecisionTreeClassifier()
model_scores(DT_model)


# In[17]:


from sklearn.model_selection import GridSearchCV
params = {'criterion':['gini','entropy'],'splitter':['best','random'],'max_leaf_nodes':list(range(2,10)),'max_features':['auto', 'sqrt', 'log2'],'min_impurity_decrease':[0.2,0.5,0.7]}
grid_model = GridSearchCV(estimator = DT_model,
                        param_grid = params,
                        scoring = 'accuracy', 
                        cv = 5)
model_score_grid(grid_model)


# In[74]:


DT_model_grid = tree.DecisionTreeClassifier(criterion= 'gini', max_features= 'auto', splitter='best')
model_scores(DT_model_grid)


# <h2 style='color:blue' align="center">RANDOM FOREST</h2>

# In[20]:


from sklearn.ensemble import RandomForestClassifier
random_model=RandomForestClassifier(n_estimators=31,class_weight='balanced')
model_scores(random_model)


# In[ ]:


from sklearn.model_selection import GridSearchCV
params = {'n_estimators':list(range(50,200)),'criterion':['gini','entropy'],'max_leaf_nodes':list(range(2,10)),'max_features':['auto', 'sqrt', 'log2'],'min_impurity_decrease':[0.2,0.5,0.7]}
grid_model = GridSearchCV(estimator = random_model,
                        param_grid = params,
                        scoring = 'accuracy', 
                        cv = 5)
model_score_grid(grid_model)


# <h2 style='color:blue' align="center">Naive Bayes</h2>

# In[91]:


from sklearn.naive_bayes import GaussianNB
naiveb_model = GaussianNB()
model_scores(naiveb_model)


# <h2 style='color:blue' align="center">Gradient Boosting</h2>

# In[77]:


from sklearn.ensemble import GradientBoostingClassifier
gradB_model = GradientBoostingClassifier(n_estimators=100, random_state=0)
model_scores(gradB_model)


# In[ ]:


from sklearn.model_selection import GridSearchCV
params = {'loss':['deviance', 'exponential'],'learning_rate':[0.1,0.5,0.7,0.9],'n_estimators':list(range(50,60)),'criterion':['friedman_mse', 'mse', 'mae'],'max_depth':list(range(1,10)),'max_features':['auto', 'sqrt', 'log2']}
grid_model = GridSearchCV(estimator = gradB_model,
                        param_grid = params,
                        scoring = 'accuracy', 
                        cv = 5)
model_score_grid(grid_model)


# <h2 style='color:blue' align="center">XGBoost</h2>

# In[78]:


from xgboost import XGBClassifier
XGB_model=XGBClassifier()
model_scores(XGB_model)


# <h2 style='color:blue' align="center">ADA BOOST</h2>

# In[80]:


from sklearn.ensemble import AdaBoostClassifier
adaB_model = AdaBoostClassifier(n_estimators=100, random_state=0)
model_scores(adaB_model)


# <h2 style='color:blue' align="center">CAT BOOST</h2>
# 

# In[92]:


from catboost import CatBoostClassifier
catB_model = CatBoostClassifier(verbose=0, n_estimators=100)
model_scores(catB_model)


# <h2 style='color:blue' align="center">LIGHT GBM </h2>

# In[82]:


from lightgbm import LGBMClassifier
lightgbm_model = LGBMClassifier( n_estimators=100)
model_scores(lightgbm_model)


# In[ ]:




