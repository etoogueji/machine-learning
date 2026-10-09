import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Create Synthetic Customer Churn Dataset
np.random.seed(42)
n = 200

tenure = np.random.randint(1, 72, n)          # Months with company
monthly_charges = np.random.uniform(20, 120, n)

# Ground truth probability of Churn (1 = Churned, 0 = Retained)
z = -0.05 * tenure + 0.03 * monthly_charges - 1.0
prob = 1 / (1 + np.exp(-z))
churn = (prob > 0.5).astype(int)

df = pd.DataFrame({'tenure': tenure, 'monthly_charges': monthly_charges, 'churn': churn})

# 2. Train / Test Split
X = df[['tenure', 'monthly_charges']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict Classes and Probabilities
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (Churn)

# 5. Evaluate Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 6. Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Retained (0)', 'Churned (1)'], 
            yticklabels=['Retained (0)', 'Churned (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix - Customer Churn')
plt.show()