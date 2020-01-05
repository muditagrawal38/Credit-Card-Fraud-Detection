
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV , train_test_split
from tqdm import tqdm_notebook
import warnings
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
import gc
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='whitegrid', palette='inferno', font_scale=1.5)


# In[3]:


dataset = pd.read_csv('creditcard.csv')


# In[4]:


dataset.head()


# In[5]:


dataset.describe()


# In[6]:


sns.heatmap(dataset)


# In[45]:


fraud = dataset[dataset['Class'] == 1]
normal= dataset[dataset['Class'] == 0]
fraud.shape, normal.shape


# ### No missing values

# In[66]:


fig,axes = plt.subplots(nrows=2,ncols=1, sharex= True, figsize = (11,7))
fig.suptitle('Distribution of time for both cases')
sns.distplot(fraud['Time'], ax = axes[0], bins = 60)
axes[0].set_title("Fraud Cases")
sns.distplot(normal['Time'], ax = axes[1], bins = 60)
axes[1].set_title("Normal Cases")


# ### We can see that fraud & normal cases cannot be separated on the basis of time

# In[67]:


sns.distplot(np.log1p(dataset['Amount']))


# ### The distribution of Amount is right skewed. Transformation required!!!

# In[68]:


fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize = (11,7), sharex= True)
ax1.hist(fraud['Amount'], bins = 60)
ax1.set_title("Fraud Cases")
ax2.hist(normal['Amount'], bins= 60)
ax2.set_title("Normal Cases")
plt.xlim(0,20000)
plt.yscale('log')


# ### Amount is smaller for fradulent cases

# In[69]:


n_data = dataset.drop(['Class', 'Time'], axis = 1)
ss = StandardScaler()
scaled_data=  pd.DataFrame(ss.fit_transform(n_data), columns = n_data.columns)


# In[80]:


sns.distplot(np.log1p(scaled_data.iloc[:,28]))


# ### The dataset has been standardized.

# In[71]:


# This is the standardized data set.
scaled_data['Class'] = dataset['Class']
X = scaled_data.drop('Class', axis= 1)
y = scaled_data['Class']


# In[72]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)


# In[78]:


sns.countplot(y_train, palette= 'Set1')


# ### The dataset is imbalanced!!!

# In[74]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(ratio = 'minority', random_state= 101)
X_sm, y_sm = sm.fit_sample(X_train,y_train)


# In[77]:


sns.countplot(x = y_sm, palette= 'Set1')


# In[71]:


def score(model, test = X_test, y_true = y_test):
    
    pred = model.predict(test)

    print('Average precision-recall score RF:\t', round(average_precision_score(y_true, pred),4)*100)
    print('\n')
    print("Area Under ROC Curve:\t",round(roc_auc_score(y_true,pred),4)*100)
    print('\n')
    print(classification_report(y_true,pred))
    precision, recall, _ = precision_recall_curve(y_true, pred)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision_score(y_true, pred)))
    
    
    
#     fpr_rf, tpr_rf, _ = roc_curve(y_true, pred)
#     roc_auc_rf = auc(fpr_rf, tpr_rf)
#     plt.figure(figsize=(8,8))
#     plt.xlim([-0.01, 1.00])
#     plt.ylim([-0.01, 1.01])
#     plt.step(fpr_rf, tpr_rf, lw=1, label='{} curve (AUC = {:0.2f})'.format('RF',roc_auc_rf))
#     #plt.fill_between(fpr_rf, tpr_rf, step='post', alpha=0.2, color='b')


#     plt.xlabel('False Positive Rate', fontsize=16)
#     plt.ylabel('True Positive Rate', fontsize=16)
#     plt.title('ROC curve', fontsize=16)
#     plt.legend(loc='lower right', fontsize=13)
#     plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
#     plt.axes().set_aspect('equal')
#     plt.show()


# ## Logistic Regression

# In[53]:


from sklearn.linear_model import LogisticRegression


# In[86]:


lr_model = LogisticRegression(random_state= 101, max_iter= 500)


# In[87]:


lr_model.fit(X_sm,y_sm)


# In[88]:


score(lr_model)


# ## DecisionTreeClassifier

# In[66]:


from sklearn.tree import DecisionTreeClassifier


# In[67]:


dctree = DecisionTreeClassifier(random_state= 101)


# In[69]:


dctree.fit(X_train, y_train)


# In[73]:


score(dctree)


# ## RandomForest

# In[74]:


from sklearn.ensemble import RandomForestClassifier


# In[77]:


rf = RandomForestClassifier(n_estimators= 500,verbose=1,n_jobs = -1, random_state= 101)


# In[78]:


rf.fit(X_train,y_train)


# In[79]:


score(rf)


# ## XGboost

# In[80]:


from xgboost import XGBClassifier


# In[81]:


xgb  = XGBClassifier(random_state = 101, n_jobs= -1)


# In[82]:


xgb.fit(X_train,y_train)


# In[84]:


score(xgb)


# ### Till now best Precision Recall Score of ~74 & Area Under ROC Curve ~ 89
# ### Lets try preparing train data using sampling

# In[82]:


fraud_cases = scaled_data[scaled_data['Class'] == 1]
normal_cases = scaled_data[scaled_data['Class'] == 0]


# In[83]:


normal_cases = normal_cases.sample(n = len(fraud_cases)*3,random_state= 101)


# In[84]:


resampled = pd.concat([fraud_cases, normal_cases])


# In[85]:


sns.countplot(resampled['Class'], palette= 'Set1')


# In[93]:


X = scaled_data.drop('Class', axis= 1)
y = scaled_data['Class']


# In[94]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)


# In[95]:


rf.fit(X_train, y_train)


# In[96]:


score(rf)


# ### Achieved a Average precision-recall score = 92.22 & Area Under ROC Curve: 96.87
