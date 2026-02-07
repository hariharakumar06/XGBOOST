import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('milk_quality_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

print("\nFirst few rows of the dataset:")
print(df.head())

print("\nMissing values in the dataset:")
print(df.isnull().sum())
print("\nDataset info:")
print(df.info())

grade_mapping = {'low': 0, 'medium': 1, 'high': 2}
df['grade'] = df['grade'].map(grade_mapping)
print("\nTarget variable 'grade' encoded:")
print(df['grade'].value_counts().sort_index())

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(df['ph'], bins=20, edgecolor='black', color='skyblue')
plt.xlabel('pH')
plt.ylabel('Frequency')
plt.title('Distribution of pH')

plt.subplot(1, 2, 2)
plt.hist(df['temperature'], bins=20, edgecolor='black', color='lightcoral')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.title('Distribution of Temperature')
plt.tight_layout()
plt.show()

X = df[['ph', 'temperature', 'taste', 'odor', 'fat', 'turbidity', 'colour']]
y = df['grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
gb_model.fit(X_train, y_train)

gb_accuracy = accuracy_score(y_test, gb_model.predict(X_test))
print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")

xgb_model = XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

xgb_tuned = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                          colsample_bytree=0.5, random_state=42)
xgb_tuned.fit(X_train, y_train)

xgb_tuned_accuracy = accuracy_score(y_test, xgb_tuned.predict(X_test))
print(f"\nXGBoost (Tuned) Accuracy: {xgb_tuned_accuracy:.4f}")

y_pred = xgb_tuned.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - XGBoost (Tuned)')
plt.show()

# Save the best model
dump(xgb_tuned, 'model.pkl')
print("\nModel saved to model.pkl")
