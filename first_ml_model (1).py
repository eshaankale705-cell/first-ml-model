"""
YOUR FIRST ML MODEL - Iris Flower Classification
==================================================
This script walks through the complete beginner ML workflow:
load data -> split data -> check data -> train -> predict -> evaluate.

Each "MISTAKE #" comment marks a common beginner error and shows
you how to avoid it.

Run it with:  python first_ml_model.py
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------------------
# STEP 1: Load the data
# -------------------------------------------------------------
# The Iris dataset is built into scikit-learn - no download needed.
# It has 150 flower samples, each with 4 measurements (features),
# and a label saying which of 3 species it is.
data = load_iris(as_frame=True)
X = data.data          # features: sepal length/width, petal length/width
y = data.target        # target: 0 = setosa, 1 = versicolor, 2 = virginica

print("First 5 rows of features:")
print(X.head())
print("\nTarget classes:", data.target_names)

# -------------------------------------------------------------
# STEP 2: Split into train and test sets
# -------------------------------------------------------------
# MISTAKE #1: Training and testing on the SAME data.
# If you do that, the model can "memorize" answers instead of
# learning patterns, and you'll never know it's failing until
# it meets real, unseen data.
#
# MISTAKE #2: Forgetting random_state.
# Without it, you get a different split every time you run this,
# so your results aren't reproducible and you can't debug reliably.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # keep 20% of data aside purely for testing
    random_state=42      # fixes the "shuffle" so results are repeatable
)
print(f"\nTraining rows: {len(X_train)}, Testing rows: {len(X_test)}")

# -------------------------------------------------------------
# STEP 3: Check your data BEFORE training
# -------------------------------------------------------------
# MISTAKE #3: Skipping data checks.
# Missing values or a heavily imbalanced target can silently
# wreck your model's performance without throwing an error.
print("\nMissing values per feature:")
print(X.isnull().sum())

print("\nClass balance (how many samples per species):")
print(y.value_counts())

# -------------------------------------------------------------
# STEP 4: Train the model
# -------------------------------------------------------------
# A Decision Tree is a great first model: it's easy to visualize
# and reason about (it's basically a series of yes/no questions).
model = DecisionTreeClassifier(
    max_depth=3,        # limits complexity so it doesn't overfit
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------------------------------------
# STEP 5: Make predictions on the held-out test set
# -------------------------------------------------------------
predictions = model.predict(X_test)

# -------------------------------------------------------------
# STEP 6: Evaluate the model properly
# -------------------------------------------------------------
# MISTAKE #4: Only looking at accuracy.
# Accuracy can look great even when the model is bad at one class
# (e.g., 90% accuracy but it never correctly identifies "virginica").
# classification_report shows precision, recall, and f1-score per class.
accuracy = accuracy_score(y_test, predictions)
print(f"\nOverall Accuracy: {accuracy:.2%}")

print("\nDetailed performance by class:")
print(classification_report(y_test, predictions, target_names=data.target_names))

# -------------------------------------------------------------
# STEP 7 (bonus): See which features mattered most
# -------------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nWhich measurements mattered most for the model's decisions:")
print(importances.sort_values(ascending=False))
