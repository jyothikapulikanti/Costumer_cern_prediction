#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Churn_Modelling.csv')

plt.figure(figsize=(4,4))
output_counts = df['Exited'].value_counts()
plt.pie(output_counts, labels=output_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.show()

plt.figure(figsize=(18,22))
numeric_features = ['Age','Tenure','EstimatedSalary']
for i, column in enumerate(numeric_features):
    plt.subplot(4,2, i + 1)
    sns.boxplot(x=df[column], color='skyblue', width=0.4)

plt.figure(figsize=(20,22))
for i, column in enumerate(['Geography','Gender','HasCrCard','IsActiveMember']):
    plt.subplot(4,2,i + 1)
    sns.countplot(x=df[column], data=df)

fig, axes = plt.subplots(3,2, figsize=(24,20))
sns.boxplot(data=df, y='CreditScore', x='Exited', ax=axes[0,0])
sns.boxplot(data=df, y='Age', x='Exited', ax=axes[0,1])
sns.boxplot(data=df, y='Tenure', x='Exited', ax=axes[1,0])
sns.boxplot(data=df, y='Balance', x='Exited', ax=axes[1,1])
sns.boxplot(data=df, y='EstimatedSalary', x='Exited', ax=axes[2,0])
axes[2,1].axis('off')
plt.show()

plt.figure(figsize=(12, 5))
sns.histplot(data=df, x='CreditScore', hue='Exited', kde=True)
plt.show()

bins = [0,669,739,850]
labels = ['Low','Medium','High']
df['CreditScoreGroup'] = pd.cut(df['CreditScore'], bins=bins, labels=labels, include_lowest=True)

plt.figure(figsize=(6,3))
sns.countplot(x='CreditScoreGroup', hue='Exited', data=df)
plt.show()

plt.figure(figsize=(5,5))
sns.scatterplot(x='Tenure', y='Balance', hue='Exited', data=df)
plt.show()

df['CreditUtilization'] = df['Balance'] / df['CreditScore']
df['InteractionScore'] = df['NumOfProducts'] + df['HasCrCard'] + df['IsActiveMember']
df['BalanceToSalaryRatio'] = df['Balance'] / df['EstimatedSalary']
df['CreditScoreAgeInteraction'] = df['CreditScore'] * df['Age']

plt.figure(figsize=(8,8))
sns.heatmap(df.drop(['RowNumber','CustomerId'], axis=1).corr(), annot=True, fmt='.2f')
plt.show()

cat_col = ['Geography','Gender','CreditScoreGroup']
encoder = LabelEncoder()
for column in cat_col:
    df[column] = encoder.fit_transform(df[column])

X = df.drop(['Exited','RowNumber','CustomerId','Surname'], axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaling_columns = [
    'Age','CreditScore','Balance','EstimatedSalary',
    'CreditUtilization','BalanceToSalaryRatio','CreditScoreAgeInteraction'
]

scaler = StandardScaler()
scaler.fit(X_train[scaling_columns])
X_train[scaling_columns] = scaler.transform(X_train[scaling_columns])
X_test[scaling_columns] = scaler.transform(X_test[scaling_columns])

models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'KNN': make_pipeline_imb(SMOTE(random_state=42), KNeighborsClassifier()),
    'SVM': make_pipeline_imb(SMOTE(random_state=42), SVC(probability=True, random_state=42)),
    'XGBoost': XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        random_state=42
    ),
    'Gradient Boosting': make_pipeline_imb(
        SMOTE(random_state=42),
        GradientBoostingClassifier(random_state=42)
    )
}

results_df = pd.DataFrame(columns=['Model','Accuracy','Recall','F1','ROC_AUC'])

lb = LabelBinarizer()
lb.fit(y_train)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    else:
        roc_auc = None

    results_df = results_df.append({
        'Model': name,
        'Accuracy': accuracy,
        'Recall': recall,
        'F1': f1,
        'ROC_AUC': roc_auc
    }, ignore_index=True)

results_df
