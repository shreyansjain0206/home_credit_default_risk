#!/usr/bin/env python
# coding: utf-8

# 

# In[5]:


import pandas as pd
import numpy as np


# In[6]:


app_test=pd.read_csv('application_test[1].csv')
app_test


# In[7]:


duplicated=app_test[app_test.duplicated(keep=False)]
no_of_duplicate=len(duplicated)
print(no_of_duplicate)


# In[8]:


app_train.shape


# In[ ]:





# In[9]:


app_train=pd.read_csv('application_train[1].csv')
app_train


# In[10]:


duplicated=app_train[app_train.duplicated(keep=False)]
no_of_duplicate=len(duplicated)
print(no_of_duplicate)


# In[11]:


app_train.shape


# In[ ]:





# In[12]:


bu=pd.read_csv('bureau[1].csv')
bu


# In[13]:


bu.dropna(inplace=True)
bu


# In[14]:


duplicated=bu[bu.duplicated(keep=False)]
no_of_duplicate=len(duplicated)
print(no_of_duplicate)


# In[15]:


bu.shape


# In[ ]:





# In[16]:


feature_bu=bu.groupby(['SK_ID_CURR','SK_ID_BUREAU']).agg(active_credits=('CREDIT_ACTIVE','min'),currency=('CREDIT_CURRENCY','min'),
    credit_days=('DAYS_CREDIT','sum'),max_overdue=('AMT_CREDIT_MAX_OVERDUE','max'),cnt_extension=('CNT_CREDIT_PROLONG','max'),credit_sum=('AMT_CREDIT_SUM','max')
                                                        ,type_credit=('CREDIT_TYPE','min'),amt_repay=('AMT_ANNUITY','sum'
))
feature_bu


# In[17]:


bu_bal=pd.read_csv('bureau_balance[1].csv')
bu_bal.dropna(inplace=True)
bu_bal


# In[18]:


duplicated=bu_bal[bu_bal.duplicated(keep=False)]
no_of_duplicate=len(duplicated)
print(no_of_duplicate)


# In[19]:


bu_bal.shape


# In[20]:


sample=pd.read_csv('sample_submission[1].csv')
sample.dropna(inplace=True)
sample


# In[21]:


duplicated=sample[sample.duplicated(keep=False)]
no_of_duplicate=len(duplicated)
print(no_of_duplicate)


# In[22]:


sample.shape


# In[23]:


bu_bal['STATUS'] = bu_bal['STATUS'].replace({'C': 0, 'X': -1})  # Replace 'C' with 0 and 'X' with -1
bu_bal['STATUS'] = bu_bal['STATUS'].astype(int)  # Convert to int


# In[24]:


df_bu_bal_feat = bu_bal.groupby('SK_ID_BUREAU').agg(status=("STATUS",'min'),month_balance=('MONTHS_BALANCE','sum'))
df_bu = bu.join(df_bu_bal_feat, on='SK_ID_BUREAU', how='left')
df_bu


# In[25]:


df_bu_feat = df_bu.groupby('SK_ID_CURR').agg(active_credits=('CREDIT_ACTIVE', 'min'), currency=('CREDIT_CURRENCY', 'min'),
                                             credit_days=('DAYS_CREDIT', 'sum'), max_overdue=('AMT_CREDIT_MAX_OVERDUE', 'max'),
                                             cnt_extension=('CNT_CREDIT_PROLONG', 'max'), credit_sum=('AMT_CREDIT_SUM', 'max'),
                                             type_credit=('CREDIT_TYPE', 'min'), amt_repay=('AMT_ANNUITY', 'sum'),
                                             monthly_balance=('month_balance', 'sum'), status=('status', 'min'))
df_bu_feat = df_bu_feat.reset_index()
df_bu_feat


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


credit=pd.read_csv('credit_card_balance[1].csv')
credit.dropna(inplace=True)
credit


# In[27]:


duplicated=credit[credit.duplicated(keep=False)]
no_of_duplicate=len(duplicated)
print(no_of_duplicate)


# In[28]:


credit.shape


# In[29]:


feature_credit = credit.groupby('SK_ID_CURR').agg(
    count_credit=('SK_ID_PREV', 'count'),
    sum_credit_balance=('AMT_BALANCE', 'sum'),
    mean_credit_limit=('AMT_CREDIT_LIMIT_ACTUAL', 'mean'),
    sum_transactions=('AMT_DRAWINGS_CURRENT', 'sum'),
    avg_inst_money=('AMT_INST_MIN_REGULARITY', 'mean'),
    sum_of_amt_recievabe=('AMT_TOTAL_RECEIVABLE', 'sum'),
    avg_count_drawing=('CNT_DRAWINGS_CURRENT', 'mean'),
    sum_mature_credits=('CNT_INSTALMENT_MATURE_CUM','sum')
    
    
    
)


feature_credit=feature_credit.reset_index()
feature_credit


# In[30]:


feature_credit['SK_ID_CURR'].dtype


# In[ ]:





# In[31]:


feature_credit.to_csv()


# In[32]:


hc = pd.read_csv('HomeCredit_columns_description[1].csv', encoding='latin1')
hc


# In[33]:


duplicated=app_test[app_test.duplicated(keep=False)]
no_of_duplicate=len(duplicated)
print(no_of_duplicate)


# In[34]:


hc.shape


# In[35]:


inst=pd.read_csv('installments_payments[1].csv')
inst.dropna(inplace=True)
inst


# In[36]:


duplicated=inst[inst.duplicated(keep=False)]
no_of_duplicate=len(duplicated)
print(no_of_duplicate)


# In[37]:


inst.shape


# In[38]:


feature_inst=inst.groupby('SK_ID_PREV').agg(inst_num=('NUM_INSTALMENT_NUMBER','max'),payment_date=('DAYS_ENTRY_PAYMENT','mean'),installment=('AMT_INSTALMENT','mean'),amt=('AMT_PAYMENT','sum'))
feature_inst=feature_inst.reset_index()
feature_inst


# In[ ]:





# In[39]:


pos=pd.read_csv('POS_CASH_balance[1].csv')
pos.dropna(inplace=True)
pos


# In[40]:


duplicated=pos[pos.duplicated(keep=False)]
no_of_duplicate=len(duplicated)
print(no_of_duplicate)


# In[41]:


pos.shape


# In[42]:


df_feat_pos=pos.groupby('SK_ID_PREV').agg(Balance=('MONTHS_BALANCE', 'max'),
    cnt_inst=('CNT_INSTALMENT', 'count'),
    Status_counts=('NAME_CONTRACT_STATUS', lambda x: x.value_counts().to_dict()),
    due_date=('SK_DPD', 'mean'))
df_feat_pos=df_feat_pos.reset_index()
df_feat_pos


# In[ ]:





# In[ ]:





# In[43]:


pre=pd.read_csv('previous_application[1].csv')
pre.dropna(inplace=True)
pre


# In[44]:


duplicated=pre[pre.duplicated(keep=False)]
no_of_duplicate=len(duplicated)
print(no_of_duplicate)


# In[45]:


pre.shape


# In[46]:


df_join_pre_pos = pre.merge(df_feat_pos, on='SK_ID_PREV', how='left', suffixes=('_pre', '_pos'))

df_join_pre_pos


# In[47]:


df_join_pre_inst = df_join_pre_pos.merge(feature_inst, on='SK_ID_PREV', how='left')

df_join_pre_inst


# In[48]:


df_join_pre_inst['SK_ID_PREV'].dtype


# In[49]:


weekday_mapping = {
    'MONDAY': 1,
    'TUESDAY': 2,
    'WEDNESDAY': 3,
    'THURSDAY': 4,
    'FRIDAY': 5,
    'SATURDAY': 6,
    'SUNDAY': 7
}

# Convert 'WEEKDAY_APPR_PROCESS_START' column to numeric using the mapping
df_join_pre_inst['WEEKDAY_APPR_PROCESS_START'] = df_join_pre_inst['WEEKDAY_APPR_PROCESS_START'].map(weekday_mapping)


# In[50]:


df_previous=df_join_pre_inst.groupby('SK_ID_CURR').agg(payment=('AMT_ANNUITY', 'sum'),
    type_of_loan=('NAME_CONTRACT_TYPE', 'min'),
    amount=('AMT_CREDIT', 'sum'),
    down_payment=('AMT_DOWN_PAYMENT', 'sum'),
    approval_day=('WEEKDAY_APPR_PROCESS_START', 'mean'),
    count=('cnt_inst', 'max'),
    first_due=('DAYS_FIRST_DUE', 'max'),
    termination=('DAYS_TERMINATION', 'max'),
    inst_num=('inst_num', 'max'),
    payment_date=('payment_date', 'mean'),
    installment=('installment', 'mean'),
    amt=('amt', 'sum'),
    Balance=('Balance', 'max'),
    cnt_inst=('cnt_inst', 'count'),
    
    due_date=('due_date', 'mean'))
df_previous


# In[51]:


df_bu_feat.head()


# In[52]:


df_merg_bu=app_train.merge(df_bu_feat,on='SK_ID_CURR',how='left')
df_merg_bu


# In[53]:


df_bu_feat[df_bu_feat['SK_ID_CURR']==100002]


# In[54]:


df_merg_bu.isna().sum()


# In[55]:


df_merge_pre=df_merg_bu.merge(df_previous,on='SK_ID_CURR',how='left')
df_merge_pre


# In[56]:


df_merge=df_merge_pre.merge(feature_credit,on='SK_ID_CURR',how='left')
df_merge


# In[57]:


df_merge.fillna(0, inplace=True)
df_merge


# In[71]:


features=df_merge.drop(columns=['TARGET','SK_ID_CURR'])
features


# In[59]:


target = df_merge['TARGET']
target


# In[60]:


target=pd.DataFrame(target)
target


# In[61]:


data_types = merged.dtypes
categorical_features = data_types[data_types == 'object'].index.tolist()
numerical_features = merged.select_dtypes(include='number').index.tolist()


# In[62]:


categorical_df = merged[categorical_features]
categorical_df


# In[63]:


categorical_df_encoded = pd.get_dummies(categorical_df)
categorical_df_encoded


# In[64]:


numerical_features = merged.select_dtypes(include='number')
numerical_features


# In[ ]:





# In[ ]:





# In[65]:


from sklearn.preprocessing import StandardScaler


# In[66]:



scaler = StandardScaler() 


# In[67]:



numerical_df_scaled = pd.DataFrame(scaler.fit_transform(numerical_features), columns=numerical_features.columns)
numerical_df_scaled


# In[68]:


df_scaled = pd.concat([numerical_df_scaled, categorical_df_encoded], axis=1)
df_scaled


# In[69]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[87]:


x_train, x_test, y_train, y_test = train_test_split(df_scaled, target, test_size=0.15, random_state=42)


# In[ ]:





# In[ ]:





# In[ ]:





# In[88]:


x_train.shape


# In[89]:


x_test.shape


# In[75]:


y_test = pd.DataFrame(y_test)
y_test.shape


# In[90]:


y_train = pd.DataFrame(y_train)
y_train.shape


# In[ ]:





# In[91]:


from sklearn.metrics import  recall_score


# In[92]:


from sklearn.metrics import precision_score


# In[79]:


get_ipython().system('pip install xgboost')


# In[80]:


import xgboost as xgb


# In[93]:


model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(target)),
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
)


# In[94]:


model.fit(x_train, y_train)


# In[95]:


y_pred = model.predict(x_test)


# In[97]:


import matplotlib.pyplot as plt


# In[98]:


model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(target)),
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
)
model.fit(
    x_train, y_train,
    eval_set=[(x_test, y_test)],
    verbose=False
)

# Retrieve the evaluation results during training
evals_result = model.evals_result()

# Extract the loss values from the evaluation result
train_loss = evals_result['validation_0']['mlogloss']
test_loss = evals_result['validation_0']['mlogloss']

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss', marker='o')
plt.plot(range(1, len(test_loss) + 1), test_loss, label='Validation Loss', marker='o')
plt.xlabel('Number of Rounds')
plt.ylabel('Log Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid()
plt.show()


# In[100]:


with open('model_predictions.txt', 'w') as file:
    # Iterate through the predicted values and write them to the file, one value per line
    for prediction in y_pred:
        file.write(str(prediction) + '\n')

print("Model predictions saved to 'model_predictions.txt'.")


# In[101]:


precision = precision_score(y_test, y_pred)
precision


# In[102]:


recall = recall_score(y_test, y_pred)
recall


# In[103]:


from sklearn.metrics import f1_score


# In[104]:


f1 = f1_score(y_test, y_pred)
f1


# In[105]:


from sklearn.metrics import roc_auc_score


# In[ ]:


roc_auc = roc_auc_score(y_test, y_pred)
roc_auc


# In[ ]:





# In[108]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

# Convert y_train DataFrame to a 1-dimensional array using values.ravel()
y_train = y_train.values.ravel()

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(x_test)

# Calculate precision and recall scores
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# In[109]:


np.savetxt('predicted_results.txt', y_pred)


# In[110]:


precision = precision_score(y_test, y_pred, zero_division=1)
print("Precision Score:", precision)


# In[111]:


from sklearn.metrics import roc_auc_score
y_prob = model.predict_proba(x_test)[:, 1]

# Calculate the ROC score
roc_score = roc_auc_score(y_test, y_prob)
print("ROC Score:", roc_score)


# In[112]:


from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)


# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




