# Titanic Survival Prediction 🚢

A machine learning pipeline to predict passenger survival on the Titanic.

## Approach
- **Data Cleaning:** Filled missing values for Age and Fare.
- **Feature Engineering:** Extracted `Title` from names, created `FamilySize`, and handled rare titles.
- **Preprocessing:** One-Hot Encoding for categorical variables.
- **Model:** Random Forest Classifier (200 trees, max depth 7).
- **Validation Accuracy:** ~82.7%

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python main.py`
