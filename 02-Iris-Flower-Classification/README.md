# 🌸 Iris Flower Classification Using Machine Learning

## 📌 Description
A Python machine learning project that classifies Iris flowers into three species: Setosa, Versicolor, and Virginica.

The model uses the K-Nearest Neighbors (KNN) algorithm to identify flowers based on their measurements.

## 🛠️ Technologies Used
- Python
- Scikit-learn
- K-Nearest Neighbors (KNN)

## ⚙️ How It Works
1. Loads the Iris flower dataset.
2. Splits data into training and testing sets.
3. Trains a KNN machine learning model.
4. Predicts Iris flower species.
5. Calculates test accuracy.
6. Allows users to enter their own flower measurements.

## 📊 Results
- **Test Accuracy:** 93.33%
- **Training Data:** 120 samples
- **Testing Data:** 30 samples
- **Flower Species:** 3

## 🚀 How to Run

Install the required library:

```bash
pip install scikit-learn
```

Run the program:

```bash
python iris_flower_classification.py
```

## 💻 Example Output

```text
IRIS FLOWER CLASSIFICATION

Test Accuracy: 93.33%

Flower 1: Setosa
Flower 2: Versicolor
Flower 3: Virginica

Would you like to test a flower? (yes/no): yes

Sepal length: 6.7
Sepal width: 3.0
Petal length: 5.2
Petal width: 2.3

AI Prediction: VIRGINICA
```
