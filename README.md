# Iris Flower Classification — My First ML Model

A beginner machine learning project that classifies iris flowers into one of three species based on their physical measurements. Built to learn the end-to-end ML workflow: data loading, splitting, training, and evaluation.

## What this project does

Trains a Decision Tree classifier on the classic Iris dataset (150 samples, 4 features, 3 species) and evaluates how well it predicts species from petal/sepal measurements.

## Key concepts practiced

- **Train/test splitting** — to properly evaluate model performance on unseen data instead of testing on the same data it was trained on
- **Data validation** — checking for missing values and class imbalance before training
- **Model evaluation** — using precision, recall, and F1-score (not just accuracy) to get a fuller picture of performance
- **Feature importance** — identifying which measurements actually drove the model's predictions

## Results

- Achieved 100% accuracy on the held-out test set (expected — Iris is a well-separated, "toy" dataset used for learning)
- Petal length was the most predictive feature (~93% importance), while sepal measurements had little to no impact

## Tech stack

- Python 3
- pandas
- scikit-learn

## How to run it

```bash
pip install scikit-learn pandas
python first_ml_model.py
```

Or paste the code into a [Google Colab](https://colab.research.google.com) notebook and run it — no installation needed.

## What I'd improve next

- Apply the same workflow to a messier, real-world dataset with missing values and class imbalance (e.g., Titanic survival dataset)
- Compare multiple model types (logistic regression, random forest) instead of just one
- Add cross-validation instead of a single train/test split for more robust evaluation

## Author

*[Your name here]*
