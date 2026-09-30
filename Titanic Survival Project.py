#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score) 
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib


# In[10]:


Titanic=pd.read_csv("Titanic_Dataset.csv")


# In[11]:


Titanic


# In[12]:


Titanic.info()


# In[13]:


Titanic.describe()


# In[14]:


Titanic.isnull().sum()


# In[15]:


Titanic["Age"]=Titanic["Age"].fillna(Titanic["Age"].median())


# In[16]:


Titanic.isnull().sum()


# In[17]:


Titanic["Cabin"]=Titanic["Cabin"].fillna(Titanic["Cabin"].mode()[0])
Titanic["Embarked"]=Titanic["Embarked"].fillna(Titanic["Embarked"].mode()[0])


# In[18]:


Titanic.isnull().sum()


# In[19]:


Titanic=Titanic.drop( 
['PassengerId', 'Name', 'Ticket', 'Cabin'], 
axis=1 
) 


# In[20]:


Titanic


# In[21]:


sns.countplot(data=Titanic, x='Survived') 
plt.title('Survival Distribution') 
plt.xlabel('Survived') 
plt.ylabel('Passenger Count') 
plt.show() 


# In[22]:


sns.countplot( 
data=Titanic, 
x='Sex', 
hue='Survived' 
) 
plt.title('Survival by Gender') 
plt.show()


# In[23]:


survival_gender = Titanic.groupby('Sex')['Survived'].mean() 
print(survival_gender)


# In[24]:


survival_gender.plot(kind='bar') 
plt.title('Survival Rate by Gender') 
plt.ylabel('Survival Rate') 
plt.show()


# In[25]:


sns.countplot( 
data=Titanic, 
x='Pclass', 
hue='Survived') 
plt.title('Survival by Passenger Class') 
plt.show()


# In[26]:


sns.countplot( 
data=Titanic, 
x='Pclass', 
hue='Survived' 
) 
plt.title('Survival by Passenger Class') 
plt.show() 


# In[27]:


class_survival = Titanic.groupby('Pclass')['Survived'].mean() 
print(class_survival) 
class_survival.plot(kind='bar') 
plt.title('Survival Rate by Passenger Class') 
plt.ylabel('Survival Rate') 
plt.show()


# In[28]:


sns.histplot( 
data=Titanic, 
x='Age', 
hue='Survived', 
kde=True 
) 
plt.title('Age Distribution by Survival') 
plt.show() 


# In[29]:


sns.histplot( 
data=Titanic, 
x='Fare', 
hue='Survived', 
kde=True 
) 
plt.title('Fare Distribution by Survival') 
plt.show()


# In[30]:


Titanic['FamilySize'] = Titanic['SibSp'] + Titanic['Parch'] + 1


# In[31]:


Titanic


# In[32]:


Titanic['IsAlone'] = (Titanic['FamilySize'] == 1).astype(int)


# In[33]:


Titanic


# In[34]:


Titanic= pd.get_dummies(Titanic,columns=['Sex', 'Embarked'], drop_first=True,dtype=int) 


# In[35]:


Titanic


# In[36]:


x = Titanic.drop('Survived', axis=1) 


# In[37]:


x


# In[38]:


y = Titanic['Survived']


# In[39]:


y


# In[40]:


x_train, x_test, y_train, y_test = train_test_split( 
x, 
y, 
test_size=0.20, 
random_state=42, 
stratify=y 
)


# In[41]:


print(x_train.shape)


# In[42]:


print(x_test.shape)


# In[43]:


print(y_train.shape)


# In[44]:


print(y_test.shape)


# In[45]:


scaler = StandardScaler() 
x_train_scaled = scaler.fit_transform(x_train) 
x_test_scaled = scaler.transform(x_test) 


# In[46]:


LR=LogisticRegression()


# In[47]:


LR


# In[48]:


LR.fit(x_train_scaled,y_train)


# In[49]:


log_pred =LR.predict(x_test_scaled)


# In[50]:


log_pred


# In[51]:


knn= KNeighborsClassifier(n_neighbors=5) 


# In[52]:


knn.fit( 
x_train_scaled, 
y_train 
) 


# In[53]:


knn_pred = knn.predict(x_test_scaled)


# In[54]:


knn_pred


# In[55]:


svm_model = SVC( 
kernel='rbf', 
probability=True, 
random_state=42 
) 


# In[56]:


svm_model.fit( 
x_train_scaled, 
y_train 
) 


# In[57]:


svm_pred = svm_model.predict(x_test_scaled)


# In[58]:


svm_pred


# In[59]:


def evaluate_model(name, y_test, y_pred): 

    print("\n", name) 

    print( 
        "Accuracy:", 
        accuracy_score(y_test, y_pred) 
    ) 

    print( 
        "Precision:", 
        precision_score(y_test, y_pred) 
    ) 

    print( 
        "Recall:", 
        recall_score(y_test, y_pred) 
    ) 

    print( 
        "F1 Score:", 
        f1_score(y_test, y_pred)
    )


# In[60]:


evaluate_model( 
    "Logistic Regression", 
    y_test, 
    log_pred 
) 

evaluate_model( 
    "KNN", 
    y_test, 
    knn_pred 
) 

evaluate_model( 
    "SVM", 
    y_test, 
    svm_pred 
)


# In[61]:


from sklearn.metrics import accuracy_score, precision_score, recall_score
f1_score 
results = pd.DataFrame({ 
'Model': [ 
'Logistic Regression', 
'KNN', 
'SVM' 
], 
'Accuracy': [ 
accuracy_score(y_test, log_pred), 
accuracy_score(y_test, knn_pred), 
accuracy_score(y_test, svm_pred) 
], 
'Precision': [ 
precision_score(y_test, log_pred), 
precision_score(y_test, knn_pred), 
precision_score(y_test, svm_pred) 
], 
'Recall': [ 
recall_score(y_test, log_pred), 
recall_score(y_test, knn_pred), 
recall_score(y_test, svm_pred) 
], 
'F1 Score': [ 
f1_score(y_test, log_pred), 
f1_score(y_test, knn_pred), 
f1_score(y_test, svm_pred) 
] 
}) 
print(results) 


# In[62]:


results.set_index('Model').plot( 
kind='bar', 
figsize=(10, 6) 
) 
plt.title('ML Model Performance Comparison') 
plt.ylabel('Score') 
plt.ylim(0, 1) 
plt.xticks(rotation=0) 
plt.show() 


# In[63]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
cm = confusion_matrix(y_test, log_pred) 
disp = ConfusionMatrixDisplay( 
confusion_matrix=cm 
) 
disp.plot() 
plt.title('Logistic Regression Confusion Matrix') 
plt.show() 


# In[64]:


cm = confusion_matrix(y_test, knn_pred) 
ConfusionMatrixDisplay( 
confusion_matrix=cm 
).plot() 
plt.title('KNN Confusion Matrix') 
plt.show()


# In[65]:


cm = confusion_matrix(y_test, svm_pred) 
ConfusionMatrixDisplay( 
confusion_matrix=cm 
).plot() 
plt.title('SVM Confusion Matrix') 
plt.show()


# In[66]:


log_prob = LR.predict_proba(x_test_scaled)[:, 1] 
knn_prob = knn.predict_proba(x_test_scaled)[:, 1] 
svm_prob = svm_model.predict_proba(x_test_scaled)[:, 1] 


# In[67]:


log_auc = roc_auc_score(y_test, log_prob) 
knn_auc = roc_auc_score(y_test, knn_prob) 
svm_auc = roc_auc_score(y_test, svm_prob)


# In[68]:


print("Logistic Regression AUC:", log_auc) 
print("KNN AUC:", knn_auc) 
print("SVM AUC:", svm_auc) 


# In[69]:


log_fpr, log_tpr, _ = roc_curve( 
y_test, 
log_prob 
) 
knn_fpr, knn_tpr, _ = roc_curve( 
y_test, 
knn_prob 
) 
svm_fpr, svm_tpr, _ = roc_curve( 
y_test, 
svm_prob 
) 
plt.figure(figsize=(8, 6)) 
plt.plot( 
log_fpr, 
log_tpr, 
label='Logistic Regression' 
) 
plt.plot( 
knn_fpr, 
knn_tpr, 
label='KNN' 
) 
plt.plot( 
svm_fpr, 
svm_tpr, 
label='SVM' 
) 
plt.plot( 
[0, 1], 
[0, 1], 
linestyle='--' 
) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve Comparison') 
plt.legend() 
plt.show()


# In[70]:


log_cv = cross_val_score( 
LR, 
x_train_scaled, 
y_train, 
cv=5, 
scoring='accuracy' 
) 


# In[71]:


knn_cv = cross_val_score( 
knn, 
x_train_scaled, 
y_train, 
cv=5, 
scoring='accuracy' 
) 


# In[72]:


svm_cv = cross_val_score( 
svm_model, 
x_train_scaled, 
y_train, 
cv=5, 
scoring='accuracy' 
) 


# In[73]:


print("Logistic:", log_cv.mean()) 
print("KNN:", knn_cv.mean()) 
print("SVM:", svm_cv.mean())


# In[74]:


k_values = range(1, 21) 
accuracies = [] 
for k in k_values: 
    model = KNeighborsClassifier( 
    n_neighbors=k 
    ) 
    model.fit( 
    x_train_scaled, 
    y_train 
    ) 
    pred = model.predict(x_test_scaled) 
    accuracies.append( 
    accuracy_score(y_test, pred) 
    ) 


# In[75]:


plt.plot( 
k_values, 
accuracies, 
marker='o' 
) 
plt.xlabel('K') 
plt.ylabel('Accuracy') 
plt.title('KNN: Accuracy vs K') 
plt.show() 


# In[76]:


kernels = ['linear', 'rbf', 'poly'] 
for kernel in kernels: 
    model = SVC( 
    kernel=kernel, 
    probability=True, 
    random_state=42 
    ) 
    model.fit( 
    x_train_scaled, 
    y_train 
    ) 
    pred = model.predict( 
    x_test_scaled 
    ) 
    print( 
    kernel, 
    accuracy_score(y_test, pred) 
    )


# In[77]:


new_passenger = pd.DataFrame({ 
'Pclass': [3], 
'Age': [25],
'SibSp': [0], 
'Parch': [0], 
'Fare': [8.5],
'FamilySize': [1], 
'IsAlone': [1], 
'Sex_male': [1],  
'Embarked_Q': [0], 
'Embarked_S': [1]
}) 
new_scaled = scaler.transform(new_passenger) 
prediction = svm_model.predict(new_scaled) 
if prediction[0] == 1: 
    print("Predicted: Survived") 
else: 
    print("Predicted: Did Not Survive") 


# In[78]:


joblib.dump(LR,"lregression.pkl")


# In[79]:


joblib.dump(knn,"knn.pkl")


# In[80]:


joblib.dump(svm_model,"svm_model.pkl")


# In[ ]:
joblib.dump(scaler,"scaler.pkl")




# In[ ]:





# In[ ]:





# In[ ]:




